/**
 * PFAC_AV_Gagniuc2025.cu
 *
 * Replication of:
 *   "The Aho-Corasick Paradigm in Modern Antivirus Engines:
 *    A Cornerstone of Signature-Based Malware Detection"
 *   Paul A. Gagniuc, Ionel-Bujorel Păvăloiu, Maria-Iuliana Dascălu
 *   Algorithms 2025, 18, 742. doi:10.3390/a18120742
 *
 * This implementation maps the paper's formal model to a GPU-accelerated
 * PFAC variant for real-time malware signature scanning:
 *
 *   A = (Q, Σ, δ, q0, F, f, O)   [Section 3.1]
 *   Time complexity : O(Σ|pi| + m + z)
 *   Sigma           : full byte alphabet (256 symbols, hex-encoded signatures)
 *
 * Key features:
 *   - Full 256-symbol byte alphabet (antivirus context vs DNA's 4)
 *   - Hex-encoded malware signatures (e.g. "25 26 33 2F 28")
 *   - Signature database loaded from .db file (name = hex_pattern format)
 *   - AC-Build: trie + BFS failure links + output propagation (Algorithm 1)
 *   - AC-Search on GPU: PFAC thread-per-byte parallel scan
 *   - Per-signature match reporting with byte offset
 *   - On-access scanner simulation: process input in streaming buffers
 *
 * Compilation:
 *   nvcc -O2 -arch=sm_70 -o pfac_av PFAC_AV_Gagniuc2025.cu
 *
 * Usage:
 *   ./pfac_av [target_file] [signature_db]
 *   Default: target.bin, signatures.db
 *
 * Signature .db format (Section Appendix A.1):
 *   EICAR = 58 31 35 30 21 50 25 40 41 50
 *   Trojan.GenericXYZ = 25 26 33 2F 28 32 33
 *   (one entry per line: name = space-separated hex bytes)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

/* ─────────────────────────────────────────────────────────
 * Constants
 * ───────────────────────────────────────────────────────── */
#define SIGMA             256        // Full byte alphabet (antivirus context)
#define MAX_STATES        50000      // Max FSM states for large signature DBs
#define MAX_SIGNATURES    1024       // Max signatures in the .db file
#define MAX_SIG_NAME      128        // Max length of a signature name
#define MAX_SIG_BYTES     256        // Max bytes per signature pattern
#define THREADS_PER_BLOCK 256        // Thread block size for GPU kernel
#define SCAN_BUFFER_SIZE  (1 << 20)  // 1 MB streaming scan buffer (Fig. 3B)

/* ─────────────────────────────────────────────────────────
 * Error checking
 * ───────────────────────────────────────────────────────── */
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "[CUDA ERROR] %s  at %s:%d\n",                     \
                    cudaGetErrorString(_err), __FILE__, __LINE__);              \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

/* ─────────────────────────────────────────────────────────
 * Signature entry (one record from .db file)
 * ───────────────────────────────────────────────────────── */
typedef struct {
    char name[MAX_SIG_NAME];         // e.g. "Trojan.Generic.2874"
    unsigned char bytes[MAX_SIG_BYTES];
    int  len;                        // number of bytes in the pattern
} Signature;

/* ─────────────────────────────────────────────────────────
 * AC Automaton (CPU-side)
 *   delta: state × SIGMA → next_state    (transitions / goto)
 *   f    : state → state                 (failure links)
 *   O    : state → signature_id (-1=none)(output function)
 *
 *   Paper Eq: δ(q,a) = child(q,a)  or  δ(f(q),a)  or  q0
 * ───────────────────────────────────────────────────────── */
typedef struct {
    int *delta;   // [MAX_STATES * SIGMA]  flattened
    int *f;       // [MAX_STATES]          failure function
    int *O;       // [MAX_STATES]          output function (sig id, -1 = none)
    int  numStates;
} AC_Automaton;

/* ─────────────────────────────────────────────────────────
 * AC-Build: insert one signature into the trie
 * (Paper Algorithm 1 — AC-Build, trie insertion phase)
 * ───────────────────────────────────────────────────────── */
void ac_insert(AC_Automaton *ac, const unsigned char *pattern, int len, int sigId) {
    int state = 0;
    for (int i = 0; i < len; i++) {
        int a = pattern[i]; // byte value 0-255
        int next = ac->delta[state * SIGMA + a];
        if (next == -1) {
            next = ac->numStates++;
            ac->delta[state * SIGMA + a] = next;
        }
        state = next;
    }
    /* Mark accepting state: O(q) ← O(q) ∪ {sigId} */
    if (ac->O[state] == -1) {
        ac->O[state] = sigId;
    }
}

/* ─────────────────────────────────────────────────────────
 * AC-Build: BFS to compute failure links and complete δ
 * (Paper Algorithm 1 — AC-Build, failure link phase)
 *
 * For each state q and symbol a:
 *   if child(q,a) exists → set f(child) = δ(f(q), a)
 *                        → O(child) |= O(f(child))   (output propagation)
 *   else                 → δ(q,a) = δ(f(q), a)        (bake failure path)
 * ───────────────────────────────────────────────────────── */
void ac_build_failure(AC_Automaton *ac) {
    int *queue = (int *)malloc(ac->numStates * sizeof(int));
    int head = 0, tail = 0;

    /* Root's children: f = root; missing root transitions → root */
    for (int a = 0; a < SIGMA; a++) {
        int child = ac->delta[0 * SIGMA + a];
        if (child > 0) {
            ac->f[child] = 0;
            queue[tail++] = child;
        } else {
            ac->delta[0 * SIGMA + a] = 0; // loop root → root
        }
    }

    /* BFS */
    while (head < tail) {
        int v = queue[head++];
        int fv = ac->f[v];

        /* Propagate output: O(v) inherits O(f(v)) */
        if (ac->O[v] == -1 && ac->O[fv] != -1) {
            ac->O[v] = ac->O[fv];
        }

        for (int a = 0; a < SIGMA; a++) {
            int u = ac->delta[v * SIGMA + a];
            if (u > 0) {
                /* Compute f(u) = δ(f(v), a) — already complete by BFS order */
                ac->f[u] = ac->delta[fv * SIGMA + a];
                /* Propagate output */
                if (ac->O[u] == -1 && ac->O[ac->f[u]] != -1) {
                    ac->O[u] = ac->O[ac->f[u]];
                }
                queue[tail++] = u;
            } else {
                /* Bake: δ(v, a) ← δ(f(v), a)  (PFAC failure elimination) */
                ac->delta[v * SIGMA + a] = ac->delta[fv * SIGMA + a];
            }
        }
    }
    free(queue);
}

/* ═══════════════════════════════════════════════════════════
 * GPU KERNEL: PFAC Malware Scanner
 *   One thread per byte position (tid) in the input buffer.
 *   Thread walks the AC automaton from tid forward until:
 *     - A match is found → record (offset, sig_id) and stop
 *     - State returns to root with no valid transition → stop
 *   (Paper Section 5.2: PFAC thread-per-byte mapping)
 *
 * match_offsets[i] = byte offset of match i
 * match_sigIds[i]  = which signature matched
 * matchCount       = total matches (atomic counter)
 * ═══════════════════════════════════════════════════════════ */
__global__ void pfacKernel_AV(
    const unsigned char *__restrict__ data, int n,
    const int           *__restrict__ delta,   // [numStates * SIGMA]
    const int           *__restrict__ output,  // [numStates]
    int *matchCount,
    int *match_offsets,
    int *match_sigIds,
    int  maxMatches,
    int  bufferOffset)   // byte offset of this buffer in the full file
{
    int tid = blockIdx.y * gridDim.x * blockDim.x
            + blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int state = 0;
    for (int i = tid; i < n; i++) {
        int a = (int)(unsigned char)data[i];
        state = delta[state * SIGMA + a];

        /* output[state] >= 0 means this state is an accepting state */
        if (output[state] >= 0) {
            int slot = atomicAdd(matchCount, 1);
            if (slot < maxMatches) {
                match_offsets[slot] = bufferOffset + tid; // start of match
                match_sigIds[slot]  = output[state];
            }
            break; // one match per thread (PFAC principle)
        }

        /* PFAC: thread dies if transitioned back to root with no further chance */
        if (state == 0) break;
    }
}

/* ─────────────────────────────────────────────────────────
 * Parse one hex byte from a string like "2F"
 * ───────────────────────────────────────────────────────── */
int parseHexByte(const char *s) {
    int val = 0;
    sscanf(s, "%x", &val);
    return val & 0xFF;
}

/* ─────────────────────────────────────────────────────────
 * Load signature database from .db file
 * Format: SIGNAME = HH HH HH ...
 * (Appendix A.1 of the paper)
 * Returns number of signatures loaded.
 * ───────────────────────────────────────────────────────── */
int loadSignatureDB(const char *filename, Signature *sigs, int maxSigs) {
    FILE *fp = fopen(filename, "r");
    int count = 0;

    if (!fp) {
        printf("[INFO] Signature DB '%s' not found. Using built-in demo signatures.\n\n",
               filename);
        /* Demo signatures from paper Fig. 2 / Appendix */
        const char *demoNames[] = {
            "Demo.Pattern.P1",     // P1 = 25 26 33
            "Demo.Pattern.P2",     // P2 = 3B 35 34
            "Demo.Pattern.P3",     // P3 = 33 35 3C
            "Demo.Pattern.P4",     // P4 = 7B 54 49 39
            "EICAR.Substring",     // Partial EICAR test string
            "Trojan.Generic.A",
            "Trojan.Generic.B"
        };
        unsigned char demoBytes[][8] = {
            {0x25, 0x26, 0x33},
            {0x3B, 0x35, 0x34},
            {0x33, 0x35, 0x3C},
            {0x7B, 0x54, 0x49, 0x39},
            {0x58, 0x35, 0x21, 0x50},
            {0x4D, 0x5A, 0x90, 0x00},  // MZ header (PE executable)
            {0x7F, 0x45, 0x4C, 0x46}   // ELF header
        };
        int demoLens[] = {3, 3, 3, 4, 4, 4, 4};
        int nd = 7;
        for (int i = 0; i < nd && i < maxSigs; i++) {
            strncpy(sigs[i].name, demoNames[i], MAX_SIG_NAME - 1);
            memcpy(sigs[i].bytes, demoBytes[i], demoLens[i]);
            sigs[i].len = demoLens[i];
            count++;
        }
        return count;
    }

    char line[1024];
    while (fgets(line, sizeof(line), fp) && count < maxSigs) {
        /* Skip comments and blank lines */
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;

        /* Split on '=' */
        char *eq = strchr(line, '=');
        if (!eq) continue;
        *eq = '\0';

        /* Parse name (trim whitespace) */
        char *name = line;
        while (*name == ' ' || *name == '\t') name++;
        char *nameEnd = name + strlen(name) - 1;
        while (nameEnd > name && (*nameEnd == ' ' || *nameEnd == '\t' || *nameEnd == '\n'))
            *nameEnd-- = '\0';

        strncpy(sigs[count].name, name, MAX_SIG_NAME - 1);

        /* Parse hex bytes */
        char *hexPart = eq + 1;
        char *tok = strtok(hexPart, " \t\r\n");
        int byteLen = 0;
        while (tok && byteLen < MAX_SIG_BYTES) {
            sigs[count].bytes[byteLen++] = (unsigned char)parseHexByte(tok);
            tok = strtok(NULL, " \t\r\n");
        }
        sigs[count].len = byteLen;

        if (byteLen > 0) count++;
    }
    fclose(fp);
    return count;
}

/* ─────────────────────────────────────────────────────────
 * Compute 2D grid layout
 * ───────────────────────────────────────────────────────── */
void computeGrid(int n, int threads, dim3 *grid, dim3 *block) {
    int totalBlocks = (n + threads - 1) / threads;
    block->x = threads; block->y = 1; block->z = 1;
    if (totalBlocks <= 65535) {
        grid->x = totalBlocks; grid->y = 1;
    } else {
        int gx = (int)ceil(sqrt((double)totalBlocks));
        grid->x = gx;
        grid->y = (totalBlocks + gx - 1) / gx;
    }
    grid->z = 1;
}

/* ─────────────────────────────────────────────────────────
 * MAIN
 * ───────────────────────────────────────────────────────── */
int main(int argc, char *argv[]) {
    const char *target_file = (argc > 1) ? argv[1] : "target.bin";
    const char *db_file     = (argc > 2) ? argv[2] : "signatures.db";

    printf("=== AC/PFAC Antivirus Scanner (Gagniuc et al., 2025) ===\n");
    printf("Target file    : %s\n", target_file);
    printf("Signature DB   : %s\n\n", db_file);

    /* ── Load signatures ─────────────────────────────────── */
    Signature *sigs = (Signature *)malloc(MAX_SIGNATURES * sizeof(Signature));
    int numSigs = loadSignatureDB(db_file, sigs, MAX_SIGNATURES);
    printf("[INFO] Loaded %d signature(s):\n", numSigs);
    for (int i = 0; i < numSigs; i++) {
        printf("  [%3d] %-30s  (%d bytes)  hex:", i, sigs[i].name, sigs[i].len);
        for (int b = 0; b < sigs[i].len && b < 8; b++)
            printf(" %02X", sigs[i].bytes[b]);
        if (sigs[i].len > 8) printf(" ...");
        printf("\n");
    }
    printf("\n");

    /* ── Build AC Automaton (Algorithm 1) ────────────────── */
    AC_Automaton ac;
    ac.numStates = 1; // q0 = root

    /* Allocate on heap (SIGMA=256 makes stack too large) */
    ac.delta = (int *)malloc((long)MAX_STATES * SIGMA * sizeof(int));
    ac.f     = (int *)malloc(MAX_STATES * sizeof(int));
    ac.O     = (int *)malloc(MAX_STATES * sizeof(int));
    if (!ac.delta || !ac.f || !ac.O) {
        fprintf(stderr, "[ERROR] Failed to allocate automaton memory.\n");
        return 1;
    }

    /* Initialize: all transitions -1, outputs -1, failures 0 */
    for (long i = 0; i < (long)MAX_STATES * SIGMA; i++) ac.delta[i] = -1;
    memset(ac.f, 0,  MAX_STATES * sizeof(int));
    memset(ac.O, -1, MAX_STATES * sizeof(int));

    /* AC-Build phase 1: insert all signatures into trie */
    printf("[BUILD] Inserting signatures into trie...\n");
    for (int i = 0; i < numSigs; i++) {
        ac_insert(&ac, sigs[i].bytes, sigs[i].len, i);
    }
    printf("[BUILD] Trie states after insertion: %d\n", ac.numStates);

    /* AC-Build phase 2: BFS failure links + complete delta */
    printf("[BUILD] Computing failure links (BFS)...\n");
    ac_build_failure(&ac);
    printf("[BUILD] Automaton ready. Total states: %d\n\n", ac.numStates);

    /* ── Open target file ────────────────────────────────── */
    FILE *fp = fopen(target_file, "rb");
    if (!fp) {
        printf("[ERROR] Could not open target file '%s'.\n", target_file);
        /* Create a small demo binary for testing */
        printf("[INFO] Creating demo binary 'target.bin' with embedded patterns...\n");
        fp = fopen("target.bin", "wb");
        if (fp) {
            /* Embed some of the demo signature bytes into a fake binary */
            unsigned char demo[] = {
                0x00, 0x01, 0x02, 0x03,
                0x4D, 0x5A, 0x90, 0x00,  // MZ header
                0x25, 0x26, 0x33,         // P1
                0xAA, 0xBB, 0xCC,
                0x3B, 0x35, 0x34,         // P2
                0xFF, 0xFF,
                0x7B, 0x54, 0x49, 0x39,   // P4
                0x00
            };
            fwrite(demo, 1, sizeof(demo), fp);
            fclose(fp);
            fp = fopen("target.bin", "rb");
            target_file = "target.bin";
        }
        if (!fp) { fprintf(stderr, "[ERROR] Could not create demo file.\n"); return 1; }
    }

    fseek(fp, 0, SEEK_END);
    long fileSize = ftell(fp);
    rewind(fp);
    printf("[INFO] Target file size: %ld bytes.\n\n", fileSize);

    /* ── GPU memory for the automaton ────────────────────── */
    long tableBytes = (long)ac.numStates * SIGMA * sizeof(int);
    int *d_delta, *d_output;
    CUDA_CHECK(cudaMalloc(&d_delta,  tableBytes));
    CUDA_CHECK(cudaMalloc(&d_output, ac.numStates * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_delta,  ac.delta, tableBytes,                   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output, ac.O,     ac.numStates * sizeof(int),   cudaMemcpyHostToDevice));

    /* ── Match result buffers ────────────────────────────── */
    int maxMatches = 65536;
    int *d_matchCount, *d_matchOffsets, *d_matchSigIds;
    int *h_matchOffsets = (int *)malloc(maxMatches * sizeof(int));
    int *h_matchSigIds  = (int *)malloc(maxMatches * sizeof(int));
    CUDA_CHECK(cudaMalloc(&d_matchCount,   sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_matchOffsets, maxMatches * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_matchSigIds,  maxMatches * sizeof(int)));

    /* ── Streaming buffer (Fig. 3B: sliding scanning window) */
    unsigned char *h_buf = (unsigned char *)malloc(SCAN_BUFFER_SIZE);
    unsigned char *d_buf;
    CUDA_CHECK(cudaMalloc(&d_buf, SCAN_BUFFER_SIZE));

    /* ── Timing ──────────────────────────────────────────── */
    cudaEvent_t evStart, evStop;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));
    float totalMs = 0.0f;

    int  totalMatches  = 0;
    long bytesScanned  = 0;

    printf("[SCAN] Scanning in %d-byte buffers (streaming window)...\n\n", SCAN_BUFFER_SIZE);

    /* ── Main scanning loop (on-access scanner simulation) ── */
    CUDA_CHECK(cudaMemset(d_matchCount, 0, sizeof(int)));

    while (!feof(fp)) {
        int bufLen = (int)fread(h_buf, 1, SCAN_BUFFER_SIZE, fp);
        if (bufLen == 0) break;

        CUDA_CHECK(cudaMemcpy(d_buf, h_buf, bufLen, cudaMemcpyHostToDevice));

        dim3 grid, block;
        computeGrid(bufLen, THREADS_PER_BLOCK, &grid, &block);

        CUDA_CHECK(cudaEventRecord(evStart));
        pfacKernel_AV<<<grid, block>>>(
            d_buf, bufLen,
            d_delta, d_output,
            d_matchCount, d_matchOffsets, d_matchSigIds,
            maxMatches,
            (int)bytesScanned);
        CUDA_CHECK(cudaEventRecord(evStop));
        CUDA_CHECK(cudaEventSynchronize(evStop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, evStart, evStop));
        totalMs += ms;

        /* Retrieve results */
        int bufMatches = 0;
        CUDA_CHECK(cudaMemcpy(&bufMatches,    d_matchCount,   sizeof(int),                   cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_matchOffsets, d_matchOffsets, bufMatches * sizeof(int),       cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_matchSigIds,  d_matchSigIds,  bufMatches * sizeof(int),       cudaMemcpyDeviceToHost));

        /* Print detections */
        for (int i = 0; i < bufMatches && i < maxMatches; i++) {
            int sigId = h_matchSigIds[i];
            if (sigId >= 0 && sigId < numSigs) {
                printf("  [DETECTION] Offset 0x%08X | Signature: %s\n",
                       h_matchOffsets[i], sigs[sigId].name);
            }
        }

        totalMatches += bufMatches;
        bytesScanned += bufLen;

        /* Reset match counter for next buffer */
        CUDA_CHECK(cudaMemset(d_matchCount, 0, sizeof(int)));
    }
    fclose(fp);

    /* ── Final report ────────────────────────────────────── */
    double throughput_MB = (bytesScanned / 1024.0 / 1024.0) / (totalMs / 1000.0);

    printf("\n═══════════════════════════════════════════════════════\n");
    printf("  File size scanned  : %ld bytes\n",  bytesScanned);
    printf("  Total detections   : %d\n",          totalMatches);
    printf("  Total scan time    : %.4f ms\n",     totalMs);
    printf("  Throughput         : %.2f MB/s\n",   throughput_MB);
    printf("  Automaton states   : %d\n",           ac.numStates);
    printf("  Signatures loaded  : %d\n",           numSigs);
    printf("═══════════════════════════════════════════════════════\n");

    /* ── Cleanup ─────────────────────────────────────────── */
    cudaFree(d_delta);
    cudaFree(d_output);
    cudaFree(d_matchCount);
    cudaFree(d_matchOffsets);
    cudaFree(d_matchSigIds);
    cudaFree(d_buf);
    cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);
    free(ac.delta);
    free(ac.f);
    free(ac.O);
    free(sigs);
    free(h_buf);
    free(h_matchOffsets);
    free(h_matchSigIds);

    return 0;
}

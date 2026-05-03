/**
 * ============================================================================
 * pfac_gpu.cu  —  Two-Phase PFAC, caveman edition
 * ============================================================================
 * Paper : Lin et al., "Two-Phase PFAC Algorithm for Multiple Patterns
 *         Matching on CUDA," IEEE TPDS 2023.  [Ref 10]
 *
 * CAVEMAN RULES:
 *   1. One thing per function. Function does exactly what its name says.
 *   2. No clever macros that hide what's happening.
 *   3. Every kernel parameter is spelled out — no pointer aliasing tricks.
 *   4. Memory path is explicit: build table → copy to GPU → run kernel.
 *   5. Timing is honest: wall time = wall, kernel time = kernel.
 *   6. If states > constant memory limit, we say so and use global. Done.
 *
 * Build (Linux / WSL):
 *   nvcc -O3 -arch=sm_89 pfac_gpu.cu -o pfac_gpu
 *
 * Build (Windows, MSVC mismatch):
 *   nvcc -O3 -arch=sm_89 --allow-unsupported-compiler pfac_gpu.cu -o pfac_gpu.exe
 *
 * Real profiling (run after build):
 *   ncu --metrics \
 *     sm__warps_active.avg.pct_of_peak_sustained_active,\
 *     smsp__sass_average_branch_targets_threads_uniform.pct,\
 *     l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
 *     sm__cycles_active.avg.pct_of_peak_sustained_elapsed \
 *     ./pfac_gpu > ncu_report.txt
 * ============================================================================
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ── platform timer ── */
#ifdef _WIN32
#  include <windows.h>
   static LARGE_INTEGER _freq;
   static int           _freq_ok = 0;
   typedef LARGE_INTEGER hrtimer_t;
   static void hrtimer_now(hrtimer_t *t) {
       if (!_freq_ok) { QueryPerformanceFrequency(&_freq); _freq_ok = 1; }
       QueryPerformanceCounter(t);
   }
   static double hrtimer_ms(hrtimer_t *a, hrtimer_t *b) {
       return (double)(b->QuadPart - a->QuadPart) /
              (double)_freq.QuadPart * 1000.0;
   }
#else
#  include <time.h>
   typedef struct timespec hrtimer_t;
   static void hrtimer_now(hrtimer_t *t) { clock_gettime(CLOCK_MONOTONIC, t); }
   static double hrtimer_ms(hrtimer_t *a, hrtimer_t *b) {
       return (b->tv_sec  - a->tv_sec ) * 1000.0 +
              (b->tv_nsec - a->tv_nsec) / 1e6;
   }
#endif

/* ── CUDA error check — abort on any error ── */
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _err = (call);                                          \
        if (_err != cudaSuccess) {                                          \
            fprintf(stderr, "[CUDA ERROR] %s\n  at %s line %d\n",          \
                    cudaGetErrorString(_err), __FILE__, __LINE__);          \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

/* ============================================================
 * CONSTANTS  —  all in one place, easy to change
 * ============================================================ */
#define DNA_ALPHA       4           /* A=0 C=1 G=2 T=3                    */
#define MAX_PATTERNS  512           /* hard cap on number of patterns      */
#define PAT_LEN        18           /* every k-mer is exactly 18 nt        */
#define FILTER_DEPTH    6           /* phase-1 exits early if depth < this */
#define MAX_STATES   8192           /* trie nodes (512 pats × 18 nt + 1)  */
#define MAX_MATCHES (1<<23)         /* 8M slots — more than enough         */
#define THREADS_PER_BLOCK 256

/*
 * Constant memory is 64 KB on all CUDA devices.
 * Table size = states × DNA_ALPHA × 4 bytes.
 * 64 KB / (4 × 4B) = 4096 states maximum.
 * We pick 2048 to be safe (leaves room for out[] and depth[]).
 * If automaton is bigger we fall back to global memory.
 */
#define CONST_MEM_MAX_STATES 2048
#define CONST_TABLE_SIZE     (CONST_MEM_MAX_STATES * DNA_ALPHA)

/* Dataset paths — edit these to match your filesystem */
#define PATH_EXONS  "data/genomic/knownCanonical.exonNuc.fa/knownCanonical.exonNuc.fa"
#define PATH_CHM13  "data/genomic/CHM13v2.0_genomic.fna/CHM13v2.0_genomic.fna"
#define PATH_DMEL   "data/genomic/dmel-all-aligned-r6.66.fasta/dmel-all-aligned-r6.66.fasta"
#define PATH_YEAST  "data/genomic/cere/strains/S288c/assembly/genome.fa"
#define FASTA_CAP_MB 256UL           /* cap each file at 1 GB              */

/* ============================================================
 * DATA TYPES
 * ============================================================ */
typedef struct {
    int pos;   /* position in text where match ends */
    int pat;   /* which pattern matched              */
} Match;

/*
 * One trie node.
 * trie_next[] = pure trie (before failure links) — used by PFAC kernel.
 * ac_next[]   = full AC transitions (after failure links) — used by CPU scan.
 * Both arrays are filled separately; they are NOT the same thing.
 */
typedef struct {
    int trie_next[DNA_ALPHA];  /* -1 = no child in trie                  */
    int ac_next  [DNA_ALPHA];  /* handles failure-link redirects for AC   */
    int fail;
    int output_pat;            /* -1 = no pattern ends here, else pat id  */
    int depth;
} TrieNode;

/* ============================================================
 * GLOBALS  —  automaton lives here on host
 * ============================================================ */
static TrieNode g_trie[MAX_STATES];
static int      g_num_states = 0;

static uint8_t  g_pats[MAX_PATTERNS][PAT_LEN];
static int      g_pat_len[MAX_PATTERNS];
static int      g_num_pats = 0;

static Match    g_match_buf[MAX_MATCHES];   /* reused scratch buffer       */
static Match    g_sweep_buf[MAX_MATCHES];   /* separate buf for sweep      */

/* ============================================================
 * CONSTANT MEMORY ARRAYS  (used when automaton fits)
 * ============================================================ */
__constant__ int d_const_table[CONST_TABLE_SIZE];
__constant__ int d_const_out  [CONST_MEM_MAX_STATES];
__constant__ int d_const_depth[CONST_MEM_MAX_STATES];

/* Global memory fallback pointers (used when automaton is too big) */
static int *d_glob_table = NULL;
static int *d_glob_out   = NULL;
static int *d_glob_depth = NULL;

/* Which path are we using right now? */
static int g_use_const_mem = 1;  /* 1 = constant, 0 = global fallback    */

/* ============================================================
 * DNA ENCODER
 * Returns 0-3 for A/C/G/T (upper or lower case).
 * Returns -1 for anything else (N, gaps, etc.).
 * ============================================================ */
static int encode_base(uint8_t c) {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default:             return -1;
    }
}

/* ============================================================
 * FASTA LOADER
 * Reads one FASTA file.  Skips header lines (> ...).
 * Encodes each base to 0-3; stores 0xFF for non-ACGT bases.
 * Stops reading after `cap_bytes` bytes of sequence.
 * Sets *out_len to how many bytes were read.
 * Returns malloc'd buffer (caller must free), or NULL on failure.
 * ============================================================ */
static uint8_t *load_fasta(const char *path, size_t *out_len, size_t cap_bytes) {
    *out_len = 0;

    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "[WARN] Cannot open: %s\n", path);
        return NULL;
    }

    /* Allocate worst-case buffer: whole file or cap + small margin */
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    rewind(f);
    size_t alloc = (cap_bytes > 0 && (size_t)file_size > cap_bytes)
                 ? cap_bytes + 4096
                 : (size_t)file_size + 4096;

    uint8_t *buf = (uint8_t *)malloc(alloc + 1);
    if (!buf) {
        fprintf(stderr, "[ERROR] malloc failed (%zu bytes)\n", alloc);
        fclose(f);
        return NULL;
    }

    size_t pos = 0;
    char   line[8192];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '>') continue;          /* skip FASTA headers */
        for (int i = 0; line[i] && line[i] != '\n' && line[i] != '\r'; i++) {
            if (cap_bytes > 0 && pos >= cap_bytes) goto done_loading;
            int enc = encode_base((uint8_t)line[i]);
            buf[pos++] = (enc >= 0) ? (uint8_t)enc : 0xFF;
        }
    }
done_loading:
    fclose(f);
    buf[pos] = '\0';
    *out_len = pos;
    return buf;
}

/* ============================================================
 * TRIE / AC AUTOMATON BUILD
 * ============================================================ */

/* Allocate a new trie node and return its index. */
static int trie_new_node(void) {
    if (g_num_states >= MAX_STATES) {
        fprintf(stderr, "[ERROR] MAX_STATES (%d) exceeded\n", MAX_STATES);
        exit(EXIT_FAILURE);
    }
    int id = g_num_states++;
    for (int c = 0; c < DNA_ALPHA; c++) {
        g_trie[id].trie_next[c] = -1;
        g_trie[id].ac_next[c]   = -1;
    }
    g_trie[id].fail       = 0;
    g_trie[id].output_pat = -1;
    g_trie[id].depth      = 0;
    return id;
}

/* Insert one pattern into the trie. */
static void trie_insert(const uint8_t *pat, int len, int pat_id) {
    int cur = 0;
    for (int i = 0; i < len; i++) {
        int c = (int)pat[i];
        if (c < 0 || c >= DNA_ALPHA) continue;  /* skip encoded 0xFF */
        if (g_trie[cur].trie_next[c] == -1) {
            int child = trie_new_node();
            g_trie[child].depth = g_trie[cur].depth + 1;
            g_trie[cur].trie_next[c] = child;
        }
        cur = g_trie[cur].trie_next[c];
    }
    g_trie[cur].output_pat = pat_id;
}

/*
 * Build AC failure links via BFS.
 *
 * IMPORTANT: We do this AFTER trie_insert() is done.
 * We copy trie_next[] into ac_next[] first, then let BFS fill in
 * the failure-link shortcuts ONLY in ac_next[].
 * trie_next[] is never touched again — PFAC kernels use trie_next[].
 */
static void ac_build_failure_links(void) {
    /* Step 1: copy pure trie into ac_next */
    for (int s = 0; s < g_num_states; s++) {
        for (int c = 0; c < DNA_ALPHA; c++) {
            g_trie[s].ac_next[c] = g_trie[s].trie_next[c];
        }
    }

    /* Step 2: BFS from root to fill failure links */
    int queue[MAX_STATES];
    int head = 0, tail = 0;

    /* Root's children: failure = root, add to queue */
    for (int c = 0; c < DNA_ALPHA; c++) {
        int s = g_trie[0].ac_next[c];
        if (s == -1) {
            g_trie[0].ac_next[c] = 0;  /* missing root child loops back to root */
        } else {
            g_trie[s].fail = 0;
            queue[tail++] = s;
        }
    }

    while (head < tail) {
        int v = queue[head++];

        /* Propagate output along suffix links */
        if (g_trie[v].output_pat == -1) {
            g_trie[v].output_pat = g_trie[g_trie[v].fail].output_pat;
        }

        for (int c = 0; c < DNA_ALPHA; c++) {
            int u = g_trie[v].ac_next[c];
            if (u == -1) {
                /* No child — follow failure link to get transition */
                g_trie[v].ac_next[c] = g_trie[g_trie[v].fail].ac_next[c];
            } else {
                /* Has child — its failure is failure[v]'s transition on c */
                int f = g_trie[g_trie[v].fail].ac_next[c];
                if (f == u) f = 0;  /* avoid self-loop */
                g_trie[u].fail = f;
                queue[tail++] = u;
            }
        }
    }
}

/* ============================================================
 * UPLOAD PFAC TABLE TO GPU
 *
 * Uploads trie_next[] (NOT ac_next[]) to GPU memory.
 * If states fit in constant memory: use cudaMemcpyToSymbol.
 * Otherwise: allocate global memory and copy there.
 *
 * Call this once per automaton build before any pfac_scan().
 * ============================================================ */
static void upload_pfac_table_to_gpu(void) {
    int ns = g_num_states;

    if (ns <= CONST_MEM_MAX_STATES) {
        /* ── Constant memory path ── */
        g_use_const_mem = 1;

        /* Build flat host arrays */
        int h_table[CONST_TABLE_SIZE];
        int h_out  [CONST_MEM_MAX_STATES];
        int h_depth[CONST_MEM_MAX_STATES];

        for (int s = 0; s < ns; s++) {
            h_out  [s] = g_trie[s].output_pat;
            h_depth[s] = g_trie[s].depth;
            for (int c = 0; c < DNA_ALPHA; c++) {
                h_table[s * DNA_ALPHA + c] = g_trie[s].trie_next[c];
            }
        }

        /* Copy to GPU constant memory */
        CUDA_CHECK(cudaMemcpyToSymbol(d_const_table, h_table,
                   ns * DNA_ALPHA * sizeof(int)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_const_out, h_out,
                   ns * sizeof(int)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_const_depth, h_depth,
                   ns * sizeof(int)));

    } else {
        /* ── Global memory fallback ── */
        g_use_const_mem = 0;
        fprintf(stderr,
                "[WARN] States (%d) > CONST_MEM_MAX_STATES (%d): "
                "using global memory.\n", ns, CONST_MEM_MAX_STATES);

        /* Free any old allocations */
        if (d_glob_table) { cudaFree(d_glob_table); d_glob_table = NULL; }
        if (d_glob_out)   { cudaFree(d_glob_out);   d_glob_out   = NULL; }
        if (d_glob_depth) { cudaFree(d_glob_depth); d_glob_depth = NULL; }

        /* Allocate and fill host-side flat arrays */
        int *h_table = (int *)malloc(ns * DNA_ALPHA * sizeof(int));
        int *h_out   = (int *)malloc(ns * sizeof(int));
        int *h_depth = (int *)malloc(ns * sizeof(int));
        if (!h_table || !h_out || !h_depth) {
            fprintf(stderr, "[ERROR] malloc failed in upload_pfac_table_to_gpu\n");
            exit(EXIT_FAILURE);
        }

        for (int s = 0; s < ns; s++) {
            h_out  [s] = g_trie[s].output_pat;
            h_depth[s] = g_trie[s].depth;
            for (int c = 0; c < DNA_ALPHA; c++) {
                h_table[s * DNA_ALPHA + c] = g_trie[s].trie_next[c];
            }
        }

        /* Copy to GPU global memory */
        CUDA_CHECK(cudaMalloc(&d_glob_table, ns * DNA_ALPHA * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_glob_out,   ns * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_glob_depth, ns * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_glob_table, h_table,
                   ns * DNA_ALPHA * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_glob_out,   h_out,
                   ns * sizeof(int),            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_glob_depth,  h_depth,
                   ns * sizeof(int),            cudaMemcpyHostToDevice));

        free(h_table);
        free(h_out);
        free(h_depth);
    }
}

/* ============================================================
 * PFAC KERNELS
 *
 * Two-Phase approach (Lin et al. 2023):
 *   Phase 1 (Filter):  each thread at position tid walks the trie.
 *                      If it reaches depth >= FILTER_DEPTH, it sets
 *                      d_flag[tid] = 1 (candidate). Otherwise exits.
 *   Phase 2 (Verify):  only threads with d_flag[tid] == 1 continue.
 *                      They walk the trie again fully to check for a
 *                      complete pattern match.
 *
 * Each thread starts at its OWN position in the text (PFAC rule).
 * Threads are independent — no communication needed.
 * __ldg() loads text data through L1 texture cache (read-only).
 * ============================================================ */

/* ── Phase 1 Filter — constant memory version ── */
__global__ void kernel_phase1_const(
    const uint8_t * __restrict__ d_text,
    int                          text_len,
    uint8_t *                    d_flag)          /* output: 1 = candidate */
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= text_len) return;

    d_flag[tid] = 0;

    int state = 0;
    int end   = tid + PAT_LEN;
    if (end > text_len) end = text_len;

    for (int i = tid; i < end; i++) {
        int c = (int)__ldg(&d_text[i]);
        if (c == 0xFF) return;              /* non-ACGT base: dead end */
        int next_state = d_const_table[state * DNA_ALPHA + c];
        if (next_state < 0) return;         /* no trie edge: dead end  */
        state = next_state;
        if (d_const_depth[state] >= FILTER_DEPTH) {
            d_flag[tid] = 1;               /* pass to phase 2         */
            return;
        }
    }
}

/* ── Phase 2 Verify — constant memory version ── */
__global__ void kernel_phase2_const(
    const uint8_t * __restrict__ d_text,
    int                          text_len,
    const uint8_t * __restrict__ d_flag,
    int *                        d_match_count,
    Match *                      d_matches,
    int                          max_matches)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= text_len)   return;
    if (!d_flag[tid])      return;         /* filtered out in phase 1 */

    int state = 0;
    int end   = tid + PAT_LEN;
    if (end > text_len) end = text_len;

    for (int i = tid; i < end; i++) {
        int c = (int)__ldg(&d_text[i]);
        if (c == 0xFF) return;
        int next_state = d_const_table[state * DNA_ALPHA + c];
        if (next_state < 0) return;
        state = next_state;
        if (d_const_out[state] != -1) {
            /* Found a match — grab a slot atomically */
            int slot = atomicAdd(d_match_count, 1);
            if (slot < max_matches) {
                d_matches[slot].pos = i;
                d_matches[slot].pat = d_const_out[state];
            }
            return;  /* one match per starting position is enough */
        }
    }
}

/* ── Phase 1 Filter — global memory fallback ── */
__global__ void kernel_phase1_global(
    const uint8_t * __restrict__ d_text,
    int                          text_len,
    uint8_t *                    d_flag,
    const int * __restrict__     g_table,
    const int * __restrict__     g_depth)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= text_len) return;

    d_flag[tid] = 0;
    int state = 0;
    int end   = tid + PAT_LEN;
    if (end > text_len) end = text_len;

    for (int i = tid; i < end; i++) {
        int c = (int)__ldg(&d_text[i]);
        if (c == 0xFF) return;
        int next_state = __ldg(&g_table[state * DNA_ALPHA + c]);
        if (next_state < 0) return;
        state = next_state;
        if (__ldg(&g_depth[state]) >= FILTER_DEPTH) {
            d_flag[tid] = 1;
            return;
        }
    }
}

/* ── Phase 2 Verify — global memory fallback ── */
__global__ void kernel_phase2_global(
    const uint8_t * __restrict__ d_text,
    int                          text_len,
    const uint8_t * __restrict__ d_flag,
    int *                        d_match_count,
    Match *                      d_matches,
    int                          max_matches,
    const int * __restrict__     g_table,
    const int * __restrict__     g_out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= text_len)  return;
    if (!d_flag[tid])     return;

    int state = 0;
    int end   = tid + PAT_LEN;
    if (end > text_len) end = text_len;

    for (int i = tid; i < end; i++) {
        int c = (int)__ldg(&d_text[i]);
        if (c == 0xFF) return;
        int next_state = __ldg(&g_table[state * DNA_ALPHA + c]);
        if (next_state < 0) return;
        state = next_state;
        if (__ldg(&g_out[state]) != -1) {
            int slot = atomicAdd(d_match_count, 1);
            if (slot < max_matches) {
                d_matches[slot].pos = i;
                d_matches[slot].pat = __ldg(&g_out[state]);
            }
            return;
        }
    }
}

/* ============================================================
 * CPU AC SCAN  —  single-threaded Aho-Corasick on host
 *
 * Used to: (a) measure CPU reference time for speedup calc,
 *          (b) measure per-point CPU time in scalability sweep.
 *
 * Uses ac_next[] (with failure links), NOT trie_next[].
 * Returns number of matches found.
 * ============================================================ */
static int cpu_ac_scan(const uint8_t *text, int text_len,
                       Match *out, int max_out) {
    int state = 0;
    int count = 0;
    for (int i = 0; i < text_len; i++) {
        int c = (int)text[i];
        if (c == 0xFF) { state = 0; continue; }
        if (c < 0 || c >= DNA_ALPHA) { state = 0; continue; }

        state = g_trie[state].ac_next[c];
        if (state < 0) state = 0;

        /* Walk suffix link chain to collect all outputs */
        int s = state;
        while (s > 0) {
            if (g_trie[s].output_pat != -1) {
                if (count < max_out) {
                    out[count].pos = i;
                    out[count].pat = g_trie[s].output_pat;
                }
                count++;
                break;
            }
            s = g_trie[s].fail;
        }
    }
    return count;
}

/* ============================================================
 * GPU PFAC SCAN WRAPPER
 *
 * Does exactly this:
 *   1. Upload table (rebuild each call — required for sweep)
 *   2. Allocate GPU memory
 *   3. Copy text to GPU
 *   4. Launch Phase 1 kernel, time it
 *   5. Launch Phase 2 kernel, time it
 *   6. Copy results back
 *   7. Free GPU memory
 *   8. Fill result struct
 *
 * Returns number of matches found.
 * ============================================================ */
typedef struct {
    double wall_ms;          /* total wall-clock time including memcpy    */
    double kernel_ms;        /* Phase1 + Phase2 kernel time only          */
    double phase1_ms;        /* Phase 1 alone                             */
    double phase2_ms;        /* Phase 2 alone                             */
    double throughput_mbps;  /* MB/s based on kernel time                 */
    double speedup;          /* kernel time vs CPU AC (filled by caller)  */
    size_t gpu_mem_bytes;    /* GPU memory allocated for this scan        */
    double occupancy_pct;    /* from cudaOccupancyMaxActiveBlocksPerMP    */
    int    match_count;
} ScanResult;

static int pfac_gpu_scan(const uint8_t *h_text, int text_len,
                         Match *h_out, ScanResult *result) {
    hrtimer_t wall_start, wall_end;
    hrtimer_now(&wall_start);

    /* Selalu upload tabel sebelum scan */
    upload_pfac_table_to_gpu();

    /* ── KONFIGURASI CHUNK / SLIDING WINDOW ── */
    // Batasi ukuran data di GPU maksimal 512 MB per iterasi
    const size_t CHUNK_SIZE = 512UL * 1024 * 1024; 
    size_t active_chunk_size = (text_len < CHUNK_SIZE) ? text_len : CHUNK_SIZE;

    /* ── Alokasikan GPU buffers dengan ukuran terbatas ── */
    uint8_t *d_text  = NULL;
    uint8_t *d_flag  = NULL;
    int     *d_count = NULL;
    Match   *d_matches = NULL;

    size_t text_bytes    = active_chunk_size;
    size_t flag_bytes    = active_chunk_size;
    size_t count_bytes   = sizeof(int);
    size_t matches_bytes = (size_t)MAX_MATCHES * sizeof(Match);

    CUDA_CHECK(cudaMalloc(&d_text,    text_bytes));
    CUDA_CHECK(cudaMalloc(&d_flag,    flag_bytes));
    CUDA_CHECK(cudaMalloc(&d_count,   count_bytes));
    CUDA_CHECK(cudaMalloc(&d_matches, matches_bytes));

    result->gpu_mem_bytes = text_bytes + flag_bytes + count_bytes + matches_bytes
                          + (size_t)g_num_states * (DNA_ALPHA + 2) * sizeof(int);

    /* ── Occupancy measurement ── */
    int max_blocks_p1 = 0, max_blocks_p2 = 0;
    if (g_use_const_mem) {
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &max_blocks_p1, kernel_phase1_const, THREADS_PER_BLOCK, 0));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &max_blocks_p2, kernel_phase2_const, THREADS_PER_BLOCK, 0));
    } else {
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &max_blocks_p1, kernel_phase1_global, THREADS_PER_BLOCK, 0));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &max_blocks_p2, kernel_phase2_global, THREADS_PER_BLOCK, 0));
    }
    cudaDeviceProp dp;
    CUDA_CHECK(cudaGetDeviceProperties(&dp, 0));
    double occ1 = (double)(max_blocks_p1 * THREADS_PER_BLOCK)
                / (double)dp.maxThreadsPerMultiProcessor * 100.0;
    double occ2 = (double)(max_blocks_p2 * THREADS_PER_BLOCK)
                / (double)dp.maxThreadsPerMultiProcessor * 100.0;
    result->occupancy_pct = (occ1 + occ2) / 2.0;
    if (result->occupancy_pct > 100.0) result->occupancy_pct = 100.0;

    /* ── CUDA events for kernel timing ── */
    cudaEvent_t ev_p1_start, ev_p1_end, ev_p2_start, ev_p2_end;
    CUDA_CHECK(cudaEventCreate(&ev_p1_start));
    CUDA_CHECK(cudaEventCreate(&ev_p1_end));
    CUDA_CHECK(cudaEventCreate(&ev_p2_start));
    CUDA_CHECK(cudaEventCreate(&ev_p2_end));

    /* ── LOOPING UNTUK MEMPROSES PER CHUNK ── */
    size_t offset = 0;
    int total_matches = 0;
    result->phase1_ms = 0.0;
    result->phase2_ms = 0.0;
    result->kernel_ms = 0.0;

    while (offset < (size_t)text_len) {
        // Hitung ukuran sisa data yang akan diproses
        size_t current_chunk = (size_t)text_len - offset;
        if (current_chunk > CHUNK_SIZE) {
            /* Kita sisakan overlap sebesar PAT_LEN agar k-mer 
               yang terpotong di ujung chunk tetap bisa terbaca */
            current_chunk = CHUNK_SIZE;
        }

        /* 1. Copy sebagian text ke GPU */
        CUDA_CHECK(cudaMemcpy(d_text, h_text + offset, current_chunk, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_flag,  0, current_chunk));
        CUDA_CHECK(cudaMemset(d_count, 0, count_bytes));
        CUDA_CHECK(cudaMemset(d_matches, 0, matches_bytes));

        int num_blocks = (current_chunk + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        /* 2. Jalankan Phase 1: Filter */
        CUDA_CHECK(cudaEventRecord(ev_p1_start));
        if (g_use_const_mem) {
            kernel_phase1_const<<<num_blocks, THREADS_PER_BLOCK>>>(
                d_text, (int)current_chunk, d_flag);
        } else {
            kernel_phase1_global<<<num_blocks, THREADS_PER_BLOCK>>>(
                d_text, (int)current_chunk, d_flag, d_glob_table, d_glob_depth);
        }
        CUDA_CHECK(cudaEventRecord(ev_p1_end));
        CUDA_CHECK(cudaEventSynchronize(ev_p1_end));
        CUDA_CHECK(cudaGetLastError());

        /* 3. Jalankan Phase 2: Verify */
        CUDA_CHECK(cudaEventRecord(ev_p2_start));
        if (g_use_const_mem) {
            kernel_phase2_const<<<num_blocks, THREADS_PER_BLOCK>>>(
                d_text, (int)current_chunk, d_flag, d_count, d_matches, MAX_MATCHES);
        } else {
            kernel_phase2_global<<<num_blocks, THREADS_PER_BLOCK>>>(
                d_text, (int)current_chunk, d_flag, d_count, d_matches, MAX_MATCHES,
                d_glob_table, d_glob_out);
        }
        CUDA_CHECK(cudaEventRecord(ev_p2_end));
        CUDA_CHECK(cudaEventSynchronize(ev_p2_end));
        CUDA_CHECK(cudaGetLastError());

        /* 4. Akumulasi Waktu Kernel */
        float ms_p1 = 0.0f, ms_p2 = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms_p1, ev_p1_start, ev_p1_end));
        CUDA_CHECK(cudaEventElapsedTime(&ms_p2, ev_p2_start, ev_p2_end));
        result->phase1_ms += (double)ms_p1;
        result->phase2_ms += (double)ms_p2;
        result->kernel_ms += (double)(ms_p1 + ms_p2);

        /* 5. Copy match count dan data match kembali ke Host */
        int gpu_count = 0;
        CUDA_CHECK(cudaMemcpy(&gpu_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
        
        int safe_count = (gpu_count < MAX_MATCHES) ? gpu_count : MAX_MATCHES;
        
        // Simpan match ke host buffer dengan mengoreksi offset posisi aslinya
        if (total_matches + safe_count <= MAX_MATCHES) {
            // 1. Change 'Match' to 'Match*'
            Match* temp_host_buf = (Match*)malloc(safe_count * sizeof(Match));

            // 2. Perform the copy
            CUDA_CHECK(cudaMemcpy(temp_host_buf, d_matches, safe_count * sizeof(Match), cudaMemcpyDeviceToHost));

            // 3. The loop remains the same, but now uses the pointer correctly
            for (int m = 0; m < safe_count; m++) {
                h_out[total_matches].pos = temp_host_buf[m].pos + offset; 
                h_out[total_matches].pat = temp_host_buf[m].pat;
                total_matches++;
            }

            // 4. Free the pointer
            free(temp_host_buf);
        }

        /* 6. Geser offset ke chunk berikutnya */
        if (current_chunk == CHUNK_SIZE) {
            // Geser sejauh CHUNK_SIZE dikurangi overlap k-mer agar tidak ada k-mer yang terlewat di batas chunk
            offset += (CHUNK_SIZE - PAT_LEN + 1);
        } else {
            offset += current_chunk;
        }
    }

    /* ── Free GPU memory ── */
    CUDA_CHECK(cudaFree(d_text));
    CUDA_CHECK(cudaFree(d_flag));
    CUDA_CHECK(cudaFree(d_count));
    CUDA_CHECK(cudaFree(d_matches));
    CUDA_CHECK(cudaEventDestroy(ev_p1_start));
    CUDA_CHECK(cudaEventDestroy(ev_p1_end));
    CUDA_CHECK(cudaEventDestroy(ev_p2_start));
    CUDA_CHECK(cudaEventDestroy(ev_p2_end));

    hrtimer_now(&wall_end);
    result->wall_ms         = hrtimer_ms(&wall_start, &wall_end);
    result->throughput_mbps = (double)text_len / 1e6
                            / (result->kernel_ms / 1000.0);
    result->match_count     = total_matches;
    return total_matches;
}

/* ============================================================
 * BUILD AUTOMATON FROM SCRATCH
 *
 * We call this once for the full dataset, then again for each
 * scalability sweep point with a subset of patterns.
 * ============================================================ */
static void build_automaton(int num_pats) {
    g_num_states = 0;
    trie_new_node();  /* create root = node 0 */
    for (int i = 0; i < num_pats; i++) {
        trie_insert(g_pats[i], g_pat_len[i], i);
    }
    ac_build_failure_links();
}

/* ============================================================
 * PRINT SEPARATOR
 * ============================================================ */
static void print_line(char c, int w) {
    for (int i = 0; i < w; i++) putchar(c);
    putchar('\n');
}

/* ============================================================
 * MAIN
 * ============================================================ */
int main(void) {
    print_line('=', 72);
    printf("  PFAC GPU  —  Two-Phase PFAC (caveman edition)\n");
    printf("  Ref: Lin et al., IEEE TPDS 2023 [Ref 10]\n");
    print_line('=', 72);
    printf("\n");

    /* ── Init CUDA ── */
    printf("Initializing CUDA...\n");
    cudaDeviceProp dp;
    CUDA_CHECK(cudaGetDeviceProperties(&dp, 0));
    printf("  GPU: %s (Compute %d.%d, %d SMs, %.1f GB)\n\n",
           dp.name, dp.major, dp.minor,
           dp.multiProcessorCount, (double)dp.totalGlobalMem / 1e9);

    /* ── Load FASTA datasets ── */
    printf("Loading FASTA datasets (cap: %lu MB each)...\n", FASTA_CAP_MB);

    size_t  len[4] = {0, 0, 0, 0};
    uint8_t *raw[4];

    const char *paths[4] = { PATH_EXONS, PATH_CHM13, PATH_DMEL, PATH_YEAST };
    const char *labels[4] = {
        "[1/4] Human exons (hg38)    ",
        "[2/4] Human T2T CHM13       ",
        "[3/4] D. melanogaster r6.66 ",
        "[4/4] S. cerevisiae S288c   "
    };
    size_t cap[4] = {
        FASTA_CAP_MB * 1024 * 1024,
        FASTA_CAP_MB * 1024 * 1024,
        FASTA_CAP_MB * 1024 * 1024,
        0   /* yeast is small, load all of it */
    };

    for (int i = 0; i < 4; i++) {
        printf("  %s... ", labels[i]); fflush(stdout);
        raw[i] = load_fasta(paths[i], &len[i], cap[i]);
        printf("%.3f MB\n", (double)len[i] / 1e6);
    }

    /* ── Concatenate into one buffer ── */
    size_t total_len = len[0] + len[1] + len[2] + len[3];
    if (total_len == 0) {
        fprintf(stderr, "[ERROR] No data loaded. Check file paths.\n");
        return 1;
    }

    uint8_t *text = (uint8_t *)malloc(total_len + 1);
    if (!text) { fprintf(stderr, "[ERROR] malloc failed\n"); return 1; }

    size_t wp = 0;
    for (int i = 0; i < 4; i++) {
        if (raw[i]) { memcpy(text + wp, raw[i], len[i]); wp += len[i]; free(raw[i]); }
    }
    text[wp] = '\0';

    printf("\n  Total input : %.3f MB\n", (double)total_len / 1e6);

    /* ── Extract k-mer motifs (evenly spaced across corpus) ── */
    int    stride = (int)(total_len / MAX_PATTERNS) + 1;
    g_num_pats = 0;
    for (size_t i = 0;
         i + (size_t)PAT_LEN <= total_len && g_num_pats < MAX_PATTERNS;
         i += stride)
    {
        int ok = 1;
        for (int j = 0; j < PAT_LEN; j++) {
            if (text[i + j] == 0xFF) { ok = 0; break; }
        }
        if (ok) {
            memcpy(g_pats[g_num_pats], text + i, PAT_LEN);
            g_pat_len[g_num_pats] = PAT_LEN;
            g_num_pats++;
        }
    }
    printf("  Motifs      : %d (k=%d nt, stride=%d)\n\n",
           g_num_pats, PAT_LEN, stride);

    /* ── Build full automaton ── */
    printf("Building AC trie + failure links...\n");
    build_automaton(g_num_pats);
    printf("  States: %d  |  PFAC table: %d KB (%s)\n\n",
           g_num_states,
           g_num_states * DNA_ALPHA * 4 / 1024,
           g_num_states <= CONST_MEM_MAX_STATES ? "__constant__" : "global memory");

    int text_len = (int)total_len;

    /* ── GPU warm-up (short run so driver is ready) ── */
    printf("GPU warm-up...\n");
    {
        ScanResult warmup; memset(&warmup, 0, sizeof(warmup));
        int warmup_len = (text_len < 8192) ? text_len : 8192;
        pfac_gpu_scan(text, warmup_len, g_match_buf, &warmup);
    }

    /* ── CPU reference time (full dataset) ── */
    printf("Measuring CPU AC reference time...\n"); fflush(stdout);
    hrtimer_t cpu_t0, cpu_t1;
    hrtimer_now(&cpu_t0);
    int cpu_hits = cpu_ac_scan(text, text_len, g_sweep_buf, MAX_MATCHES);
    hrtimer_now(&cpu_t1);
    double cpu_ref_ms = hrtimer_ms(&cpu_t0, &cpu_t1);
    printf("  CPU AC: %d matches in %.3f ms\n\n", cpu_hits, cpu_ref_ms);

    /* ── Main PFAC scan ── */
    printf("Running Two-Phase PFAC scan...\n"); fflush(stdout);
    ScanResult main_result;
    memset(&main_result, 0, sizeof(main_result));

    hrtimer_t scan_t0, scan_t1;
    hrtimer_now(&scan_t0);
    pfac_gpu_scan(text, text_len, g_match_buf, &main_result);
    hrtimer_now(&scan_t1);

    main_result.wall_ms = hrtimer_ms(&scan_t0, &scan_t1);
    main_result.speedup = (main_result.kernel_ms > 0)
                        ? cpu_ref_ms / main_result.kernel_ms : 0.0;

    printf("  Done: %d matches in %.3f ms wall / %.3f ms kernel  (%.2f MB/s)\n",
           main_result.match_count, main_result.wall_ms,
           main_result.kernel_ms, main_result.throughput_mbps);
    printf("  Kernel: Phase1=%.3fms  Phase2=%.3fms  Total=%.3fms\n\n",
           main_result.phase1_ms, main_result.phase2_ms, main_result.kernel_ms);

    /* ── Scalability sweep ──
     *
     * 5 points: vary both pattern count and input size simultaneously.
     * For each point:
     *   1. Rebuild automaton with subset of patterns
     *   2. Measure CPU time on that subset × input size
     *   3. Measure GPU time on same subset × input size
     *   Both use the same subset → speedup is apples-to-apples.
     */
    printf("Scalability sweep (5 points, CPU + GPU each)...\n"); fflush(stdout);

    const double pat_fracs [5] = { 0.10, 0.25, 0.50, 0.75, 1.00 };
    const double size_fracs[5] = { 0.05, 0.15, 0.30, 0.60, 1.00 };

    typedef struct {
        int    num_pats;
        size_t input_bytes;
        double cpu_ms;
        double gpu_kernel_ms;
        double throughput_mbps;
        double speedup;
    } SweepPoint;

    SweepPoint sweep[5];

    for (int si = 0; si < 5; si++) {
        int    sub_pats = (int)(g_num_pats * pat_fracs[si]);
        if (sub_pats < 1) sub_pats = 1;
        size_t sub_len  = (size_t)(total_len * size_fracs[si]);
        if (sub_len < 1024) sub_len = 1024;

        /* Rebuild automaton for this pattern subset */
        build_automaton(sub_pats);

        /* CPU time on this subset */
        hrtimer_t ct0, ct1;
        hrtimer_now(&ct0);
        cpu_ac_scan(text, (int)sub_len, g_sweep_buf, MAX_MATCHES);
        hrtimer_now(&ct1);
        double sub_cpu_ms = hrtimer_ms(&ct0, &ct1);

        /* GPU time on this subset (upload_pfac_table called inside) */
        ScanResult sub_gpu;
        memset(&sub_gpu, 0, sizeof(sub_gpu));
        pfac_gpu_scan(text, (int)sub_len, g_sweep_buf, &sub_gpu);

        sweep[si].num_pats        = sub_pats;
        sweep[si].input_bytes     = sub_len;
        sweep[si].cpu_ms          = sub_cpu_ms;
        sweep[si].gpu_kernel_ms   = sub_gpu.kernel_ms;
        sweep[si].throughput_mbps = sub_gpu.throughput_mbps;
        sweep[si].speedup         = (sub_gpu.kernel_ms > 0)
                                  ? sub_cpu_ms / sub_gpu.kernel_ms : 0.0;

        printf("  [%d/5] %3d motifs x %.2f MB : CPU=%.3fms  GPU=%.3fms  %.2fx\n",
               si + 1, sub_pats, (double)sub_len / 1e6,
               sub_cpu_ms, sub_gpu.kernel_ms, sweep[si].speedup);
        fflush(stdout);
    }

    /* Restore full automaton after sweep */
    build_automaton(g_num_pats);

    /* ============================================================
     * FINAL REPORT
     * ============================================================ */
    printf("\n");
    print_line('=', 72);
    printf("  PFAC GPU — BENCHMARK RESULTS\n");
    printf("  Ref: Lin et al., IEEE TPDS 2023 (Two-Phase PFAC) [Ref 10]\n");
    print_line('=', 72);

    printf("\n  GPU         : %s  (Compute %d.%d, %d SMs, %.1f GB)\n",
           dp.name, dp.major, dp.minor,
           dp.multiProcessorCount, (double)dp.totalGlobalMem / 1e9);
    printf("  Algorithm   : Two-Phase PFAC (Phase1=Filter, Phase2=Verify)\n");
    printf("  Alphabet    : DNA 4-symbol {A,C,G,T}\n");
    printf("  Table       : %d states x %d x 4B = %d KB (%s)\n",
           g_num_states, DNA_ALPHA, g_num_states * DNA_ALPHA * 4 / 1024,
           g_use_const_mem ? "__constant__ memory" : "global memory");
    printf("  Patterns    : %d k-mers (k=%d nt)  |  AC States: %d\n",
           g_num_pats, PAT_LEN, g_num_states);
    printf("  Input       : %.3f MB (%d bytes)\n\n",
           (double)text_len / 1e6, text_len);

    print_line('-', 72);
    printf("  COMPARISON METRICS  (compare with ac_baseline output)\n");
    print_line('-', 72);
    printf("  %-36s %12.3f ms\n", "1a. Execution Time (wall)",   main_result.wall_ms);
    printf("  %-36s %12.3f ms\n", "1b. Execution Time (kernel)",  main_result.kernel_ms);
    printf("  %-36s %12.2f MB/s\n","2. Throughput (kernel time)", main_result.throughput_mbps);
    printf("  %-36s %12.2fx\n",   "3. SpeedUp (kernel vs CPU AC)",main_result.speedup);
    printf("     CPU AC reference               :  %.3f ms\n", cpu_ref_ms);
    printf("  %-36s %12.2f MB\n", "4. Memory Usage (GPU)",
           (double)main_result.gpu_mem_bytes / 1e6);
    printf("  %-36s %12d\n",      "5. Matches Found",             main_result.match_count);
    printf("     Accuracy note                  :  PFAC finds overlapping hits (see Notes)\n");

    printf("\n  6. Scalability (GPU kernel time vs CPU AC on same subset):\n");
    printf("     %-8s  %-12s  %10s  %10s  %13s  %8s\n",
           "Motifs", "Input(MB)", "CPU(ms)", "GPU(ms)", "Throughput", "SpeedUp");
    for (int i = 0; i < 5; i++) {
        printf("     %-8d  %-12.3f  %10.3f  %10.3f  %7.1f MB/s  %7.2fx\n",
               sweep[i].num_pats,
               (double)sweep[i].input_bytes / 1e6,
               sweep[i].cpu_ms,
               sweep[i].gpu_kernel_ms,
               sweep[i].throughput_mbps,
               sweep[i].speedup);
    }

    printf("\n");
    print_line('-', 72);
    printf("  PROFILING METRICS  (Two-Phase PFAC Kernel)\n");
    print_line('-', 72);
    printf("  %-38s %10.3f ms\n", "1. Kernel Time - Phase 1 (Filter)", main_result.phase1_ms);
    printf("  %-38s %10.3f ms\n", "   Kernel Time - Phase 2 (Verify)", main_result.phase2_ms);
    printf("  %-38s %10.3f ms\n", "   Kernel Time - Total",            main_result.kernel_ms);
    printf("  %-38s %9.1f %%\n",  "2. Occupancy (CUDA API)",           main_result.occupancy_pct);
    printf("  %-38s %9s\n",       "3. Warp Efficiency",                "run ncu *");
    printf("  %-38s %9s\n",       "4. Branch Divergence",              "run ncu *");
    printf("  %-38s %9s\n",       "5. Memory Throughput",              "run ncu *");
    printf("  * Use ncu command below for real paper-ready values.\n");

    printf("\n");
    print_line('-', 72);
    printf("  REAL PROFILING COMMAND (run this for paper metrics):\n");
    print_line('-', 72);
    printf("  Linux:\n");
    printf("    ncu --metrics \\\n");
    printf("      sm__warps_active.avg.pct_of_peak_sustained_active,\\\n");
    printf("      smsp__sass_average_branch_targets_threads_uniform.pct,\\\n");
    printf("      l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\\\n");
    printf("      sm__cycles_active.avg.pct_of_peak_sustained_elapsed \\\n");
    printf("      ./pfac_gpu > ncu_report.txt\n\n");
    printf("  Windows:\n");
    printf("    ncu --metrics ^\n");
    printf("      sm__warps_active.avg.pct_of_peak_sustained_active,^\n");
    printf("      smsp__sass_average_branch_targets_threads_uniform.pct,^\n");
    printf("      l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,^\n");
    printf("      sm__cycles_active.avg.pct_of_peak_sustained_elapsed ^\n");
    printf("      pfac_gpu.exe > ncu_report.txt\n\n");
    printf("  ncu metric -> paper metric mapping:\n");
    printf("    sm__warps_active.*pct              -> Warp Efficiency\n");
    printf("    *branch_targets*uniform.pct        -> 100%% - value = Branch Div.\n");
    printf("    l1tex__t_bytes*global*ld / time    -> Memory Throughput\n");
    printf("    sm__cycles_active.*pct             -> Occupancy\n");

    printf("\n");
    print_line('-', 72);
    printf("  NOTES\n");
    print_line('-', 72);
    printf("  * PFAC: one thread per input position, pure trie traversal.\n");
    printf("  * No failure links in kernel — failure-link-free is the point.\n");
    printf("  * Phase 1 exits early if depth < %d — most threads die here.\n",
           FILTER_DEPTH);
    printf("  * Phase 2 only runs for positions that passed phase 1.\n");
    printf("  * PFAC match count can differ from AC: overlapping matches\n");
    printf("    are both correct. Compare pattern IDs, not position counts.\n");
    printf("  * SpeedUp = CPU AC wall time / GPU kernel time only.\n");
    printf("    H2D/D2H transfer excluded for fair algorithmic comparison.\n");
    print_line('=', 72);
    printf("\n");

    /* ── Cleanup ── */
    free(text);
    if (d_glob_table) cudaFree(d_glob_table);
    if (d_glob_out)   cudaFree(d_glob_out);
    if (d_glob_depth) cudaFree(d_glob_depth);

    return 0;
}

/**
 * pfac_parallel_rewrite.cu
 *
 * Clean rewrite of the GPU-parallel PFAC Aho-Corasick scanner.
 * Inspired by: Lin et al., IEEE Trans. Comput. 2012 (PFAC paradigm)
 * Extension of: Gagniuc et al., Algorithms 2025, 18, 742
 *
 * Changes vs original pfac_parallel.cu:
 *   - ALL large arrays heap-allocated (no Windows stack overflow)
 *   - Automaton uses pointer-based struct (same pattern as rewritten original)
 *   - BFS queue heap-allocated (was 40 KB stack frame)
 *   - inputStream heap-allocated (was 1 MB stack frame)
 *   - patternBytes heap-allocated
 *   - FIXED: atomicOr on unsigned long long split into two 32-bit ops
 *   - FIXED: PFAC termination condition (break on nextState==0, not pos==startPos)
 *   - FIXED: __restrict__ added to read-only kernel pointers
 *   - All allocations properly freed on exit
 *
 * Compilation (replace sm_75 with your GPU's compute capability):
 *   nvcc -O2 -arch=sm_75 -allow-unsupported-compiler -Xcompiler "/std:c++14" -o ac_parallel pfac_parallel_rewrite.cu
 *   .\ac_parallel.exe
 *
 *   GPU compute capability reference:
 *     RTX 20xx / Tesla T4  -> sm_75
 *     RTX 30xx             -> sm_86
 *     RTX 40xx             -> sm_89
 *     A100                 -> sm_80
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

/* ------------------------------------------------------------------ */
/*  Constants                                                           */
/* ------------------------------------------------------------------ */

#define ALPHABET_SIZE     256
#define MAX_STATES        10000
#define MAX_PATTERNS      64
#define MAX_PAT_LEN       128
#define MAX_INPUT_LEN     1048576   /* 1 MB */
#define THREADS_PER_BLOCK 256

/* ------------------------------------------------------------------ */
/*  CUDA error-checking macro                                           */
/* ------------------------------------------------------------------ */

#define CUDA_CHECK(call)                                                 \
    do {                                                                  \
        cudaError_t _e = (call);                                          \
        if (_e != cudaSuccess) {                                          \
            fprintf(stderr, "[CUDA ERROR] %s:%d  %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(_e));          \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

/* ------------------------------------------------------------------ */
/*  Host-side automaton — fully heap-allocated                         */
/*                                                                      */
/*  sizeof original struct was ~10 MB — never put on the stack.        */
/*  We store pointers instead and malloc each array separately.        */
/* ------------------------------------------------------------------ */

typedef struct {
    int                *delta;      /* [MAX_STATES * ALPHABET_SIZE] flattened */
    int                *f;          /* [MAX_STATES] failure links             */
    unsigned long long *O;          /* [MAX_STATES] output bitmasks           */
    int                 numStates;
} AhoCorasickAutomaton;

#define DELTA(ac, s, c)  ((ac)->delta[(s) * ALPHABET_SIZE + (c)])

static AhoCorasickAutomaton *acAlloc(void)
{
    AhoCorasickAutomaton *ac =
        (AhoCorasickAutomaton *)malloc(sizeof(AhoCorasickAutomaton));
    if (!ac) return NULL;

    ac->delta = (int *)malloc((size_t)MAX_STATES * ALPHABET_SIZE * sizeof(int));
    ac->f     = (int *)malloc((size_t)MAX_STATES * sizeof(int));
    ac->O     = (unsigned long long *)malloc((size_t)MAX_STATES * sizeof(unsigned long long));

    if (!ac->delta || !ac->f || !ac->O) {
        free(ac->delta); free(ac->f); free(ac->O); free(ac);
        return NULL;
    }
    return ac;
}

static void acFree(AhoCorasickAutomaton *ac)
{
    if (!ac) return;
    free(ac->delta); free(ac->f); free(ac->O); free(ac);
}

/* ------------------------------------------------------------------ */
/*  Utility — parse hex string to byte array                           */
/* ------------------------------------------------------------------ */

static int parseHexPattern(const char    *hexStr,
                            unsigned char *out,
                            int            maxLen)
{
    int len = 0;
    const char *p = hexStr;
    while (*p && len < maxLen) {
        while (*p == ' ') p++;
        if (!*p) break;
        unsigned int byte;
        if (sscanf(p, "%02x", &byte) == 1) {
            out[len++] = (unsigned char)byte;
            p += 2;
        } else break;
    }
    return len;
}

/* ------------------------------------------------------------------ */
/*  CPU — Trie insertion                                                */
/* ------------------------------------------------------------------ */

static void insertPattern(AhoCorasickAutomaton *ac,
                           const unsigned char  *pattern,
                           int                   patLen,
                           int                   patIndex)
{
    int state = 0;
    for (int i = 0; i < patLen; i++) {
        unsigned char c = pattern[i];
        if (DELTA(ac, state, c) == -1)
            DELTA(ac, state, c) = ac->numStates++;
        state = DELTA(ac, state, c);
    }
    ac->O[state] |= (1ULL << patIndex);
}

/* ------------------------------------------------------------------ */
/*  CPU — Failure links (BFS)                                          */
/* ------------------------------------------------------------------ */

static void buildFailureLinks(AhoCorasickAutomaton *ac)
{
    /* Heap-allocated queue — avoids 40 KB stack frame */
    int *queue = (int *)malloc((size_t)MAX_STATES * sizeof(int));
    if (!queue) { fprintf(stderr, "OOM (queue)\n"); return; }
    int head = 0, tail = 0;

    for (int c = 0; c < ALPHABET_SIZE; c++) {
        if (DELTA(ac, 0, c) != -1) {
            int s = DELTA(ac, 0, c);
            ac->f[s] = 0;
            queue[tail++] = s;
        } else {
            DELTA(ac, 0, c) = 0;
        }
    }

    while (head < tail) {
        int v = queue[head++];
        for (int c = 0; c < ALPHABET_SIZE; c++) {
            int u = DELTA(ac, v, c);
            if (u == -1) {
                DELTA(ac, v, c) = DELTA(ac, ac->f[v], c);
                continue;
            }
            ac->f[u] = DELTA(ac, ac->f[v], c);
            ac->O[u] |= ac->O[ac->f[u]];
            queue[tail++] = u;
        }
    }

    free(queue);
}

static void acBuild(AhoCorasickAutomaton     *ac,
                    const unsigned char      *patterns[],
                    const int                 patLens[],
                    int                       numPatterns)
{
    memset(ac->delta, -1, (size_t)MAX_STATES * ALPHABET_SIZE * sizeof(int));
    memset(ac->f,      0, (size_t)MAX_STATES * sizeof(int));
    memset(ac->O,      0, (size_t)MAX_STATES * sizeof(unsigned long long));
    ac->numStates = 1;

    for (int i = 0; i < numPatterns; i++)
        insertPattern(ac, patterns[i], patLens[i], i);

    buildFailureLinks(ac);
}

/* ------------------------------------------------------------------ */
/*  GPU Kernel — PFAC-style parallel scan                              */
/*                                                                      */
/*  Each thread i owns byte i of the input and runs an independent     */
/*  AC traversal from root. No failure links needed at scan time —     */
/*  the PFAC model.                                                     */
/*                                                                      */
/*  FIX 1: __restrict__ on read-only pointers enables the GPU          */
/*          read-only cache path, reducing global memory latency.       */
/*  FIX 2: termination is now `if (nextState == 0) break` — correct    */
/*          PFAC semantics. Original `pos == startPos` condition was    */
/*          too narrow and let threads spin needlessly.                 */
/*  FIX 3: atomicOr split into two 32-bit ops — CUDA has no native     */
/*          64-bit atomicOr overload, the original call was wrong.      */
/* ------------------------------------------------------------------ */

__global__ void pfacScanKernel(
    const unsigned char    * __restrict__ d_text,
    int                                   textLen,
    const int              * __restrict__ d_delta,
    const unsigned long long * __restrict__ d_output,
    unsigned long long                   *d_matches,
    int                                   numStates)
{
    int startPos = blockIdx.x * blockDim.x + threadIdx.x;
    if (startPos >= textLen) return;

    int state = 0;

    for (int pos = startPos; pos < textLen; pos++) {
        unsigned char c         = d_text[pos];
        int           nextState = d_delta[state * ALPHABET_SIZE + c];

        /* FIX 2: correct PFAC termination — stop as soon as we return
         * to root, regardless of which byte position triggered it.     */
        if (nextState == 0) break;

        state = nextState;

        if (d_output[state] != 0ULL) {
            /* FIX 3: CUDA has no 64-bit atomicOr — split into lo/hi   */
            unsigned int maskLo = (unsigned int)(d_output[state] & 0xFFFFFFFFULL);
            unsigned int maskHi = (unsigned int)(d_output[state] >> 32);
            if (maskLo) atomicOr((unsigned int *)&d_matches[pos],       maskLo);
            if (maskHi) atomicOr(((unsigned int *)&d_matches[pos]) + 1, maskHi);
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Host — flatten delta table for GPU upload                          */
/* ------------------------------------------------------------------ */

static void flattenDelta(const AhoCorasickAutomaton *ac,
                          int                        *flat,
                          int                         numStates)
{
    /* ac->delta is already flat (1-D) in the rewritten struct */
    memcpy(flat, ac->delta,
           (size_t)numStates * ALPHABET_SIZE * sizeof(int));
}

/* ------------------------------------------------------------------ */
/*  Host — launch parallel scan and collect results                    */
/* ------------------------------------------------------------------ */

static int parallelACScan(const AhoCorasickAutomaton *ac,
                           const unsigned char        *text,
                           int                         textLen,
                           const char                 *patternNames[],
                           int                         numPatterns)
{
    printf("\n[PFAC-Scan] Launching GPU parallel scan on %d bytes...\n", textLen);

    /* ---- Flatten delta for device upload ----------------------- */
    size_t deltaSize = (size_t)ac->numStates * ALPHABET_SIZE * sizeof(int);
    int *h_delta = (int *)malloc(deltaSize);
    if (!h_delta) { fprintf(stderr, "OOM (h_delta)\n"); return -1; }
    flattenDelta(ac, h_delta, ac->numStates);

    /* ---- Device allocations ------------------------------------ */
    unsigned char      *d_text    = NULL;
    int                *d_delta   = NULL;
    unsigned long long *d_output  = NULL;
    unsigned long long *d_matches = NULL;

    CUDA_CHECK(cudaMalloc(&d_text,    (size_t)textLen       * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_delta,   deltaSize));
    CUDA_CHECK(cudaMalloc(&d_output,  (size_t)ac->numStates * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_matches, (size_t)textLen       * sizeof(unsigned long long)));

    /* ---- Copy to device --------------------------------------- */
    CUDA_CHECK(cudaMemcpy(d_text,   text,     (size_t)textLen * sizeof(unsigned char),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_delta,  h_delta,  deltaSize,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output, ac->O,    (size_t)ac->numStates * sizeof(unsigned long long),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_matches, 0,       (size_t)textLen * sizeof(unsigned long long)));

    /* ---- Kernel launch ---------------------------------------- */
    int blocks = (textLen + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    printf("[PFAC-Scan] Grid: %d blocks x %d threads  (%d total threads)\n",
           blocks, THREADS_PER_BLOCK, blocks * THREADS_PER_BLOCK);

    cudaEvent_t tStart, tStop;
    CUDA_CHECK(cudaEventCreate(&tStart));
    CUDA_CHECK(cudaEventCreate(&tStop));
    CUDA_CHECK(cudaEventRecord(tStart));

    pfacScanKernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_text, textLen, d_delta, d_output, d_matches, ac->numStates);

    CUDA_CHECK(cudaEventRecord(tStop));
    CUDA_CHECK(cudaEventSynchronize(tStop));
    CUDA_CHECK(cudaGetLastError());

    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, tStart, tStop));
    printf("[PFAC-Scan] Kernel execution time: %.3f ms\n", elapsedMs);

    /* ---- Copy results back ------------------------------------ */
    unsigned long long *h_matches =
        (unsigned long long *)calloc(textLen, sizeof(unsigned long long));
    if (!h_matches) { fprintf(stderr, "OOM (h_matches)\n"); return -1; }
    CUDA_CHECK(cudaMemcpy(h_matches, d_matches,
                          (size_t)textLen * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));

    /* ---- Report matches --------------------------------------- */
    int matchCount = 0;
    for (int i = 0; i < textLen; i++) {
        if (h_matches[i] != 0ULL) {
            for (int p = 0; p < numPatterns; p++) {
                if (h_matches[i] & (1ULL << p)) {
                    printf("  [MATCH] Pattern %-20s found ending at byte offset %d\n",
                           patternNames[p], i);
                    matchCount++;
                }
            }
        }
    }
    printf("[PFAC-Scan] Done. Total matches: %d\n", matchCount);

    /* ---- Cleanup ---------------------------------------------- */
    free(h_delta);
    free(h_matches);
    cudaFree(d_text);
    cudaFree(d_delta);
    cudaFree(d_output);
    cudaFree(d_matches);
    cudaEventDestroy(tStart);
    cudaEventDestroy(tStop);

    return matchCount;
}

/* ------------------------------------------------------------------ */
/*  Main                                                                */
/* ------------------------------------------------------------------ */

int main(void)
{
    printf("========================================================\n");
    printf("  Aho-Corasick Parallel (GPU/PFAC) — Clean Rewrite      \n");
    printf("  Inspired by: Lin et al., IEEE Trans. Comput. 2012     \n");
    printf("  Extension of: Gagniuc et al., Algorithms 2025, 18,742 \n");
    printf("========================================================\n\n");

    /* ---- GPU info --------------------------------------------- */
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s | SM count: %d | Global mem: %.1f GB\n\n",
           prop.name, prop.multiProcessorCount,
           (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    /* ---- Signatures ------------------------------------------- */
    const char *sigNames[] = {
        "P1=25_26_33",
        "P2=3B_35_34",
        "P3=33_35_3C",
        "P4=7B_54_49_39"
    };
    const char *sigHex[] = {
        "25 26 33",
        "3B 35 34",
        "33 35 3C",
        "7B 54 49 39"
    };
    int numSigs = 4;

    /* patternBytes on heap */
    unsigned char (*patternBytes)[MAX_PAT_LEN] =
        (unsigned char (*)[MAX_PAT_LEN])malloc(MAX_PATTERNS * MAX_PAT_LEN);
    const unsigned char *patterns[MAX_PATTERNS];
    int patLens[MAX_PATTERNS];

    if (!patternBytes) { fprintf(stderr, "OOM\n"); return 1; }

    for (int i = 0; i < numSigs; i++) {
        patLens[i]  = parseHexPattern(sigHex[i], patternBytes[i], MAX_PAT_LEN);
        patterns[i] = patternBytes[i];
        printf("Loaded signature [%s] (%d bytes)\n", sigNames[i], patLens[i]);
    }

    /* ---- Build automaton -------------------------------------- */
    AhoCorasickAutomaton *ac = acAlloc();
    if (!ac) { fprintf(stderr, "OOM\n"); free(patternBytes); return 1; }

    printf("\n[AC-Build] Constructing automaton (CPU)...\n");
    acBuild(ac, patterns, patLens, numSigs);
    printf("[AC-Build] Done. Total states: %d\n", ac->numStates);

    /* ---- Input stream ----------------------------------------- */
    const char *inputHex =
        "33 3B 3B 35 34 35 38 35 3B 37 45 32 44 3B 25 26 33";

    unsigned char *inputStream = (unsigned char *)malloc(MAX_INPUT_LEN);
    if (!inputStream) {
        fprintf(stderr, "OOM\n");
        acFree(ac); free(patternBytes); return 1;
    }
    int inputLen = parseHexPattern(inputHex, inputStream, MAX_INPUT_LEN);
    printf("\nInput stream (%d bytes): %s\n", inputLen, inputHex);

    /* ---- GPU scan --------------------------------------------- */
    parallelACScan(ac, inputStream, inputLen, sigNames, numSigs);

    /* ---- Cleanup ---------------------------------------------- */
    free(inputStream);
    free(patternBytes);
    acFree(ac);

    printf("\n[Done] Parallel GPU-PFAC scan complete.\n");
    return 0;
}

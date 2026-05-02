/*
 * ============================================================================
 * ac_dna.cu  —  Aho-Corasick for DNA Sequence Matching on CUDA
 * ============================================================================
 * BASELINE:
 *   Gagniuc, P.A.; Păvăloiu, I.-B.; Dascălu, M.-I.
 *   "The Aho-Corasick Paradigm in Modern Antivirus Engines: A Cornerstone of
 *    Signature-Based Malware Detection."
 *   Algorithms 2025, 18, 742. https://doi.org/10.3390/a18120742
 *   GitHub: https://github.com/Gagniuc/Aho-Corasick-Native-Malware-Scanner
 *
 * ALGORITHM (Algorithm 1 from paper):
 *   AC-Build: trie insertion + BFS failure links + output propagation
 *   AC-Search: single-pass linear traversal with failure link fallback
 *
 * ADAPTATION:
 *   - Alphabet restricted to DNA: {A, C, G, T, N} => mapped to 0-4
 *   - AC automaton flattened to 2D array for GPU global memory
 *   - Each CUDA thread processes one chunk of the text (data-parallel AC / DPAC)
 *   - Boundary overlap handled by extending each chunk by (max_pattern_len - 1)
 *
 * METRICS COLLECTED:
 *   Comparison : Execution Time, Throughput, SpeedUp, Memory Usage, Accuracy, Scalability
 *   Profiling  : Kernel Time (cudaEvent), Occupancy (theoretical via API),
 *                Warp Efficiency, Branch Divergence, Memory Throughput
 *                (use nvprof / Nsight Compute for full profiling metrics)
 *
 * BUILD:
 *   nvcc -O2 -arch=sm_75 -o ac_dna ac_dna.cu
 *
 * RUN:
 *   ./ac_dna <text_file.fa> <patterns_file.txt>
 *
 * PATTERN FILE FORMAT (one pattern per line, DNA bases only):
 *   ACGT
 *   GATTACA
 *   TTAGGG
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda_occupancy.h>

/* ── constants ─────────────────────────────────────────────────────────────── */
#define ALPHA       5          /* A C G T N                                    */
#define MAX_STATES  500000     /* max automaton states                         */
#define MAX_PATTERNS 100000    /* max number of patterns                       */
#define MAX_PAT_LEN  1024      /* max single pattern length                    */
#define CHUNK_SIZE   (1 << 20) /* 1 MB per thread chunk (adjustable)           */
#define BLOCK_SIZE   256       /* CUDA threads per block                       */

/* ── DNA alphabet mapping ───────────────────────────────────────────────────
 * A→0, C→1, G→2, T→3, N→4, everything else → -1 (skip)                     */
__host__ __device__ int dna_char(char c)
{
    switch (c | 32) {   /* tolower */
        case 'a': return 0;
        case 'c': return 1;
        case 'g': return 2;
        case 't': return 3;
        case 'n': return 4;
        default:  return -1;
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 * HOST-SIDE AC AUTOMATON  (Algorithm 1, Gagniuc et al. 2025)
 * ══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    int  go[ALPHA];    /* goto transitions δ(q, a)                            */
    int  fail;         /* failure link f(q)                                   */
    int  output;       /* pattern id if accepting, else -1                    */
    int  out_via_fail; /* chained output through failure links                */
} ACState;

typedef struct {
    ACState *states;
    int      num_states;
    int      num_patterns;
    int      max_pat_len;
} ACAuto;

/* ── AC-Build (Algorithm 1 — Build phase) ─────────────────────────────────
 * Step 1: insert all patterns into trie
 * Step 2: BFS to set failure links + propagate outputs                      */
void ac_build(ACAuto *ac, char **patterns, int n_pat)
{
    ac->states = (ACState *)calloc(MAX_STATES, sizeof(ACState));
    ac->num_states = 1;   /* root = state 0 */
    ac->num_patterns = n_pat;
    ac->max_pat_len = 0;

    /* initialise root */
    memset(&ac->states[0], -1, sizeof(ACState));
    ac->states[0].fail = 0;
    ac->states[0].output = -1;
    ac->states[0].out_via_fail = -1;

    /* ── Step 1: trie insertion ─────────────────────────────────────────── */
    for (int p = 0; p < n_pat; p++) {
        int cur = 0;
        int len = (int)strlen(patterns[p]);
        if (len > ac->max_pat_len) ac->max_pat_len = len;

        for (int i = 0; i < len; i++) {
            int a = dna_char(patterns[p][i]);
            if (a < 0) continue;

            if (ac->states[cur].go[a] == -1) {
                int ns = ac->num_states++;
                memset(&ac->states[ns], -1, sizeof(ACState));
                ac->states[ns].fail = 0;
                ac->states[ns].output = -1;
                ac->states[ns].out_via_fail = -1;
                ac->states[cur].go[a] = ns;
            }
            cur = ac->states[cur].go[a];
        }
        /* mark accepting state with pattern id */
        ac->states[cur].output = p;
    }

    /* ── Step 2: BFS failure links (Algorithm 1 — AC-Build) ────────────── */
    int *queue = (int *)malloc(MAX_STATES * sizeof(int));
    int head = 0, tail = 0;

    /* depth-1 children: failure → root */
    for (int a = 0; a < ALPHA; a++) {
        int s = ac->states[0].go[a];
        if (s == -1) {
            ac->states[0].go[a] = 0;   /* optional root self-loop             */
        } else {
            ac->states[s].fail = 0;
            queue[tail++] = s;
        }
    }

    while (head < tail) {
        int v = queue[head++];
        /* propagate output via failure link (Algorithm 1: O(u) ← O(u)∪O(f(u))) */
        int fv = ac->states[v].fail;
        if (ac->states[fv].output != -1)
            ac->states[v].out_via_fail = fv;
        else
            ac->states[v].out_via_fail = ac->states[fv].out_via_fail;

        for (int a = 0; a < ALPHA; a++) {
            int u = ac->states[v].go[a];
            if (u == -1) {
                /* complete transition table — δ(v,a) = δ(f(v),a) */
                ac->states[v].go[a] = ac->states[fv].go[a];
            } else {
                /* compute failure link for u */
                int x = fv;
                while (ac->states[x].go[a] == -1 && x != 0)
                    x = ac->states[x].fail;
                int fu = ac->states[x].go[a];
                ac->states[u].fail = (fu == u) ? 0 : fu;
                queue[tail++] = u;
            }
        }
    }
    free(queue);
}

/* flatten automaton → 2D int array for GPU transfer
 * layout: go_table[state * ALPHA + a] = next_state               */
int *ac_flatten(ACAuto *ac)
{
    int *tbl = (int *)malloc((size_t)ac->num_states * ALPHA * sizeof(int));
    for (int s = 0; s < ac->num_states; s++)
        for (int a = 0; a < ALPHA; a++)
            tbl[s * ALPHA + a] = (ac->states[s].go[a] == -1) ? 0
                                                               : ac->states[s].go[a];
    return tbl;
}

/* output table: output_tbl[s] = pattern_id or -1 */
int *ac_output_table(ACAuto *ac)
{
    int *out = (int *)malloc(ac->num_states * sizeof(int));
    for (int s = 0; s < ac->num_states; s++)
        out[s] = ac->states[s].output;
    return out;
}

/* ══════════════════════════════════════════════════════════════════════════
 * GPU KERNEL  —  Data-Parallel AC (DPAC with boundary overlap)
 * Each thread processes one chunk of size CHUNK_SIZE + max_pat_len - 1.
 * This is the standard GPU adaptation of Algorithm 1 AC-Search.
 * ══════════════════════════════════════════════════════════════════════════ */
__global__ void ac_kernel(
    const char  * __restrict__ text,          /* full DNA text on device     */
    long long                  text_len,
    const int   * __restrict__ go_table,      /* [num_states * ALPHA]        */
    const int   * __restrict__ out_table,     /* [num_states]                */
    int                        num_states,
    int                        max_pat_len,
    long long   * __restrict__ match_count,   /* total matches (atomic)      */
    int                        chunk_sz)
{
    long long tid    = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long start  = tid * chunk_sz;
    if (start >= text_len) return;

    long long end = start + chunk_sz + max_pat_len - 1;
    if (end > text_len) end = text_len;

    int state = 0;
    long long local_matches = 0;

    /* AC-Search (Algorithm 1 — Search phase): single-pass traversal */
    for (long long i = start; i < end; i++) {
        int a = dna_char(text[i]);
        if (a < 0) { state = 0; continue; }

        /* δ(state, a) — pre-computed complete transition table (no failure loop) */
        state = go_table[state * ALPHA + a];

        if (out_table[state] != -1)
            local_matches++;
    }

    if (local_matches > 0)
        atomicAdd(match_count, local_matches);
}

/* ══════════════════════════════════════════════════════════════════════════
 * CPU REFERENCE AC-Search  (for accuracy verification & speedup baseline)
 * ══════════════════════════════════════════════════════════════════════════ */
long long ac_search_cpu(ACAuto *ac, const char *text, long long len)
{
    long long matches = 0;
    int state = 0;
    for (long long i = 0; i < len; i++) {
        int a = dna_char(text[i]);
        if (a < 0) { state = 0; continue; }
        state = ac->states[state].go[a];
        if (state == -1) state = 0;
        if (ac->states[state].output != -1) matches++;
        /* also follow out_via_fail chain */
        int tmp = ac->states[state].out_via_fail;
        while (tmp != -1) {
            matches++;
            tmp = ac->states[tmp].out_via_fail;
        }
    }
    return matches;
}

/* ══════════════════════════════════════════════════════════════════════════
 * UTILITIES
 * ══════════════════════════════════════════════════════════════════════════ */
static char **load_patterns(const char *fname, int *n_out)
{
    FILE *f = fopen(fname, "r");
    if (!f) { fprintf(stderr, "Cannot open pattern file: %s\n", fname); exit(1); }

    char **pats = (char **)malloc(MAX_PATTERNS * sizeof(char *));
    char  buf[MAX_PAT_LEN];
    int   n = 0;
    while (fgets(buf, sizeof(buf), f) && n < MAX_PATTERNS) {
        int L = strlen(buf);
        while (L > 0 && (buf[L-1] == '\n' || buf[L-1] == '\r' || buf[L-1] == ' '))
            buf[--L] = 0;
        if (L == 0) continue;
        /* skip FASTA header lines */
        if (buf[0] == '>') continue;
        pats[n] = strdup(buf);
        n++;
    }
    fclose(f);
    *n_out = n;
    return pats;
}

static char *load_text(const char *fname, long long *len_out)
{
    FILE *f = fopen(fname, "rb");
    if (!f) { fprintf(stderr, "Cannot open text file: %s\n", fname); exit(1); }
    fseek(f, 0, SEEK_END);
    long long sz = ftell(f);
    rewind(f);
    char *buf = (char *)malloc(sz + 1);
    fread(buf, 1, sz, f);
    buf[sz] = 0;
    fclose(f);
    *len_out = sz;
    return buf;
}

static void check_cuda(cudaError_t e, const char *msg)
{
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error [%s]: %s\n", msg, cudaGetErrorString(e));
        exit(1);
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ══════════════════════════════════════════════════════════════════════════ */
int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <dna_text.fa> <patterns.txt>\n", argv[0]);
        return 1;
    }

    /* ── load input ──────────────────────────────────────────────────────── */
    long long text_len;
    char *text = load_text(argv[1], &text_len);

    int    n_pat;
    char **patterns = load_patterns(argv[2], &n_pat);
    printf("=== AC DNA String Matching (CUDA) ===\n");
    printf("Text size   : %.2f MB (%lld bytes)\n", text_len/1e6, text_len);
    printf("Patterns    : %d\n", n_pat);

    /* ── build automaton (Algorithm 1 AC-Build) ──────────────────────────── */
    ACAuto ac;
    ac_build(&ac, patterns, n_pat);
    printf("AC states   : %d\n", ac.num_states);
    printf("Max pat len : %d\n\n", ac.max_pat_len);

    /* ── flatten for GPU ─────────────────────────────────────────────────── */
    int *go_tbl  = ac_flatten(&ac);
    int *out_tbl = ac_output_table(&ac);

    /* ── memory usage report ─────────────────────────────────────────────── */
    size_t go_bytes  = (size_t)ac.num_states * ALPHA * sizeof(int);
    size_t out_bytes = (size_t)ac.num_states * sizeof(int);
    size_t txt_bytes = (size_t)text_len;
    printf("[METRIC] Memory Usage (host automaton) : %.2f MB\n", (go_bytes+out_bytes)/1e6);
    printf("[METRIC] Memory Usage (text on GPU)    : %.2f MB\n\n", txt_bytes/1e6);

    /* ── CPU reference (for accuracy + speedup baseline) ───────────────── */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    long long cpu_matches = ac_search_cpu(&ac, text, text_len);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double cpu_time_s = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    printf("[METRIC] CPU Execution Time : %.4f s\n", cpu_time_s);
    printf("[METRIC] CPU Matches        : %lld\n\n", cpu_matches);

    /* ── GPU setup ───────────────────────────────────────────────────────── */
    char *d_text;
    int  *d_go, *d_out;
    long long *d_matches, h_matches = 0;

    check_cuda(cudaMalloc(&d_text,    txt_bytes),              "malloc text");
    check_cuda(cudaMalloc(&d_go,      go_bytes),               "malloc go");
    check_cuda(cudaMalloc(&d_out,     out_bytes),              "malloc out");
    check_cuda(cudaMalloc(&d_matches, sizeof(long long)),      "malloc cnt");
    check_cuda(cudaMemcpy(d_text,    text,    txt_bytes,  cudaMemcpyHostToDevice), "cp text");
    check_cuda(cudaMemcpy(d_go,      go_tbl,  go_bytes,   cudaMemcpyHostToDevice), "cp go");
    check_cuda(cudaMemcpy(d_out,     out_tbl, out_bytes,  cudaMemcpyHostToDevice), "cp out");
    check_cuda(cudaMemcpy(d_matches, &h_matches, sizeof(long long),
                          cudaMemcpyHostToDevice), "cp cnt");

    /* ── kernel launch config ────────────────────────────────────────────── */
    int chunk_sz   = CHUNK_SIZE;
    long long n_chunks = (text_len + chunk_sz - 1) / chunk_sz;
    int n_blocks   = (int)((n_chunks + BLOCK_SIZE - 1) / BLOCK_SIZE);

    /* ── theoretical occupancy ───────────────────────────────────────────── */
    int max_active_blocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks, ac_kernel, BLOCK_SIZE, 0);
    int device_id;
    cudaGetDevice(&device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    float occupancy = (max_active_blocks * BLOCK_SIZE) /
                      (float)prop.maxThreadsPerMultiProcessor;
    printf("[METRIC] Theoretical Occupancy : %.2f %%\n", occupancy * 100.0f);
    printf("[METRIC] SM count              : %d\n\n", prop.multiProcessorCount);

    /* ── kernel timing ───────────────────────────────────────────────────── */
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    cudaEventRecord(ev_start);
    ac_kernel<<<n_blocks, BLOCK_SIZE>>>(
        d_text, text_len, d_go, d_out,
        ac.num_states, ac.max_pat_len,
        d_matches, chunk_sz);
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    check_cuda(cudaGetLastError(), "kernel launch");

    float kernel_ms;
    cudaEventElapsedTime(&kernel_ms, ev_start, ev_stop);
    double gpu_time_s = kernel_ms / 1000.0;

    /* ── results ─────────────────────────────────────────────────────────── */
    check_cuda(cudaMemcpy(&h_matches, d_matches, sizeof(long long),
                          cudaMemcpyDeviceToHost), "cp result");

    double throughput_gbps = (text_len / 1e9) / gpu_time_s;
    double speedup         = cpu_time_s / gpu_time_s;
    double accuracy        = (cpu_matches > 0)
                             ? (double)h_matches / cpu_matches * 100.0
                             : (h_matches == 0 ? 100.0 : 0.0);

    printf("=== METRICS SUMMARY ===\n");
    printf("[METRIC] Kernel Time (GPU)       : %.4f ms\n",   kernel_ms);
    printf("[METRIC] GPU Execution Time      : %.4f s\n",    gpu_time_s);
    printf("[METRIC] Throughput              : %.4f GB/s\n", throughput_gbps);
    printf("[METRIC] SpeedUp (GPU vs CPU)    : %.2f x\n",   speedup);
    printf("[METRIC] GPU Matches             : %lld\n",      h_matches);
    printf("[METRIC] CPU Matches (reference) : %lld\n",      cpu_matches);
    printf("[METRIC] Accuracy                : %.2f %%\n",   accuracy);
    printf("\n[NOTE] For Warp Efficiency, Branch Divergence, Memory Throughput:\n");
    printf("       Run: nsys profile --stats=true ./ac_dna <text> <patterns>\n");
    printf("       or : nv-nsight-cu-cli --metrics \\\n");
    printf("              smsp__sass_average_branch_targets_threads_uniform.pct,\\\n");
    printf("              smsp__warps_active.avg.pct_of_peak_sustained_active,\\\n");
    printf("              l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum \\\n");
    printf("              ./ac_dna <text> <patterns>\n\n");

    /* ── scalability note ────────────────────────────────────────────────── */
    printf("[METRIC] Scalability: run with increasing text sizes (VDB → CHM13_rna\n");
    printf("         → CHM13_genomic → T.aestivum) and record Execution Time + Throughput.\n\n");

    /* ── cleanup ─────────────────────────────────────────────────────────── */
    cudaFree(d_text); cudaFree(d_go); cudaFree(d_out); cudaFree(d_matches);
    cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
    free(go_tbl); free(out_tbl);
    free(ac.states);
    free(text);
    for (int i = 0; i < n_pat; i++) free(patterns[i]);
    free(patterns);

    return 0;
}

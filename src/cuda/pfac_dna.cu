/*
 * ============================================================================
 * pfac_dna.cu  —  Two-Phase PFAC for DNA Sequence Matching on CUDA
 * ============================================================================
 * BASELINE:
 *   Lai, W.-S.; Wu, C.-C.; Lai, L.-F.; Sie, M.-C.
 *   "Two-Phase PFAC Algorithm for Multiple Patterns Matching on CUDA GPUs."
 *   Electronics 2019, 8, 270. https://doi.org/10.3390/electronics8030270
 *
 * ALGORITHM (Figures 5, 7, 8 from paper):
 *   Phase 1 (Figure 5): Initial State Traversal
 *     - Each thread starts at its position in text
 *     - First 2 chars → Prefix PFAC Table (shared memory, 128×128)
 *     - Remaining chars up to THRESHOLD → Suffix PFAC Table (global memory)
 *     - Threads not done at threshold → marked in Incomplete[]
 *   Job Compression (Figure 7): prefix-sum + thread remapping
 *     - Active (incomplete) threads compacted into front threads of block
 *   Phase 2 (Figure 8): Remainder Traversal
 *     - Only remapped threads continue matching from saved state
 *
 * DNA ADAPTATION:
 *   - Alphabet: {A, C, G, T, N} → 5 symbols (index 0-4)
 *   - Prefix table: 5×5 = 25 entries (fits trivially in shared memory)
 *   - Suffix table: num_states × 5 integers in global memory
 *   - Pattern IDs: states 1..n_pat are final states (PFAC convention)
 *   - Initial state: n_pat + 1  (= INITIAL_STATE in paper)
 *   - Trap state: 0             (= TRAP_STATE in paper)
 *
 * METRICS COLLECTED:
 *   Comparison : Execution Time, Throughput, SpeedUp, Memory Usage, Accuracy, Scalability
 *   Profiling  : Kernel Time (cudaEvent), Occupancy, Warp Efficiency,
 *                Branch Divergence, Memory Throughput
 *                (full profiling: nsys / Nsight Compute)
 *
 * BUILD:
 *   nvcc -O2 -arch=sm_75 -o pfac_dna pfac_dna.cu
 *
 * RUN:
 *   ./pfac_dna <text_file.fa> <patterns_file.txt>
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuda_occupancy.h>

/* ── Cross-platform timer ───────────────────────────────────────────────────── */
#ifdef _WIN32
#  include <windows.h>
   typedef LARGE_INTEGER timespec_t;
   static LARGE_INTEGER _qpf;
   static inline void timer_init(){ QueryPerformanceFrequency(&_qpf); }
   static inline void clock_now(timespec_t *t){ QueryPerformanceCounter(t); }
   static inline double elapsed_sec(timespec_t *a, timespec_t *b){
       return (double)(b->QuadPart - a->QuadPart) / (double)_qpf.QuadPart;
   }
#else
#  include <time.h>
   typedef struct timespec timespec_t;
   static inline void timer_init(){}
   static inline void clock_now(timespec_t *t){ clock_gettime(CLOCK_MONOTONIC, t); }
   static inline double elapsed_sec(timespec_t *a, timespec_t *b){
       return (b->tv_sec - a->tv_sec) + (b->tv_nsec - a->tv_nsec)*1e-9;
   }
#endif

/* ── constants ─────────────────────────────────────────────────────────────── */
#define ALPHA         5          /* A C G T N                                   */
#define MAX_STATES    500000     /* max PFAC automaton states                   */
#define MAX_PATTERNS  100000     /* max patterns                                */
#define MAX_PAT_LEN   1024       /* max single pattern length                   */
#define BLOCK_SIZE    1024       /* threads per block (paper uses 1024)         */
#define THRESHOLD     5          /* phase-1 max transitions (paper: Table 1&2)  */
#define TRAP_STATE    0          /* mismatch / dead state                       */

/* ── DNA alphabet ──────────────────────────────────────────────────────────── */
__host__ __device__ int dna_char(unsigned char c)
{
    switch (c | 32) {
        case 'a': return 0;
        case 'c': return 1;
        case 'g': return 2;
        case 't': return 3;
        case 'n': return 4;
        default:  return -1;   /* non-DNA byte → skip                          */
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 * HOST-SIDE PFAC AUTOMATON CONSTRUCTION
 *
 * PFAC convention (Lai et al., Section 1 / Lin et al. 2013):
 *   States 1 .. n_pat       : final states (one per pattern)
 *   State  n_pat+1           : initial state (INITIAL_STATE)
 *   States n_pat+2 ..        : internal transition states
 *   State  0                 : trap state (TRAP_STATE)
 *
 * No failure links are stored — the automaton is failure-link-free.
 * A thread terminates immediately when it reaches TRAP_STATE.
 * ══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    int go[ALPHA];     /* next state; 0 = TRAP_STATE                          */
    int is_final;      /* 1 if accepting state                                */
    int pattern_id;    /* 0-based pattern id if final, else -1                */
} PFACState;

typedef struct {
    PFACState *states;
    int        num_states;
    int        initial_state;   /* n_pat + 1                                  */
    int        n_pat;
    int        max_pat_len;
} PFACAuto;

void pfac_build(PFACAuto *pfac, char **patterns, int n_pat)
{
    pfac->states = (PFACState *)calloc(MAX_STATES, sizeof(PFACState));
    pfac->n_pat  = n_pat;

    /* States 0 (trap) and 1..n_pat (final) pre-allocated */
    /* State n_pat+1 = initial state */
    pfac->initial_state = n_pat + 1;
    pfac->num_states    = n_pat + 2;   /* 0 (trap), 1..n_pat (finals), n_pat+1 (init) */
    pfac->max_pat_len   = 0;

    /* initialise trap state */
    for (int a = 0; a < ALPHA; a++) pfac->states[TRAP_STATE].go[a] = TRAP_STATE;
    pfac->states[TRAP_STATE].is_final   = 0;
    pfac->states[TRAP_STATE].pattern_id = -1;

    /* initialise initial state */
    for (int a = 0; a < ALPHA; a++)
        pfac->states[pfac->initial_state].go[a] = TRAP_STATE;
    pfac->states[pfac->initial_state].is_final   = 0;
    pfac->states[pfac->initial_state].pattern_id = -1;

    /* insert each pattern into PFAC trie (failure-free, from initial state) */
    for (int p = 0; p < n_pat; p++) {
        int cur = pfac->initial_state;
        int len = (int)strlen(patterns[p]);
        if (len > pfac->max_pat_len) pfac->max_pat_len = len;

        for (int i = 0; i < len; i++) {
            int a = dna_char((unsigned char)patterns[p][i]);
            if (a < 0) continue;

            if (pfac->states[cur].go[a] == TRAP_STATE) {
                if (i == len - 1) {
                    /* last char → transition to final state for this pattern */
                    int final_id = p + 1;   /* final states are 1..n_pat      */
                    pfac->states[cur].go[a]          = final_id;
                    pfac->states[final_id].is_final   = 1;
                    pfac->states[final_id].pattern_id = p;
                    for (int b = 0; b < ALPHA; b++)
                        if (pfac->states[final_id].go[b] == 0)
                            pfac->states[final_id].go[b] = TRAP_STATE;
                } else {
                    int ns = pfac->num_states++;
                    for (int b = 0; b < ALPHA; b++)
                        pfac->states[ns].go[b] = TRAP_STATE;
                    pfac->states[ns].is_final   = 0;
                    pfac->states[ns].pattern_id = -1;
                    pfac->states[cur].go[a]     = ns;
                    cur = ns;
                }
            } else {
                cur = pfac->states[cur].go[a];
                if (i == len - 1) {
                    /* pattern ends at existing state — mark as final */
                    pfac->states[cur].is_final   = 1;
                    pfac->states[cur].pattern_id = p;
                }
            }
        }
    }
}

/* ── Flatten state transition table for GPU ─────────────────────────────────
 * suffix_table[state * ALPHA + a] = next_state
 * This is the "Suffix PFAC Table" in global memory (Lai et al., Section 3.2) */
int *pfac_flatten_suffix(PFACAuto *pfac)
{
    int *tbl = (int *)malloc((size_t)pfac->num_states * ALPHA * sizeof(int));
    for (int s = 0; s < pfac->num_states; s++)
        for (int a = 0; a < ALPHA; a++)
            tbl[s * ALPHA + a] = pfac->states[s].go[a];
    return tbl;
}

/* ── Prefix PFAC Table ───────────────────────────────────────────────────────
 * Encodes transitions for the FIRST TWO characters from the initial state.
 * prefix_table[a1 * ALPHA + a2] = state after consuming (a1, a2).
 * Size: ALPHA × ALPHA = 25 ints → fits in shared memory easily.             */
int *pfac_build_prefix_table(PFACAuto *pfac)
{
    int *tbl = (int *)calloc(ALPHA * ALPHA, sizeof(int));
    for (int a1 = 0; a1 < ALPHA; a1++) {
        int s1 = pfac->states[pfac->initial_state].go[a1];
        for (int a2 = 0; a2 < ALPHA; a2++) {
            if (s1 == TRAP_STATE)
                tbl[a1 * ALPHA + a2] = TRAP_STATE;
            else
                tbl[a1 * ALPHA + a2] = pfac->states[s1].go[a2];
        }
    }
    return tbl;
}

/* ── is_final table ─────────────────────────────────────────────────────────
 * is_final[s] == 1 → s < INITIAL_STATE (i.e., s is a pattern-id state)     */
int *pfac_final_table(PFACAuto *pfac)
{
    int *tbl = (int *)calloc(pfac->num_states, sizeof(int));
    for (int s = 0; s < pfac->num_states; s++)
        tbl[s] = pfac->states[s].is_final;
    return tbl;
}

/* ══════════════════════════════════════════════════════════════════════════
 * DEVICE HELPERS
 * ══════════════════════════════════════════════════════════════════════════ */

/* parallel prefix-sum (scan) within a block — used for job compression
 * Adapted from: Harris, Sengupta & Owens, GPU Gems 3, Chapter 39
 * (cited as [37] in Lai et al.)                                              */
__device__ void block_prefix_sum(volatile int *data, int tid, int n)
{
    int offset = 1;
    /* up-sweep */
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2*tid+1) - 1;
            int bi = offset * (2*tid+2) - 1;
            data[bi] += data[ai];
        }
        offset <<= 1;
    }
    /* clear last element */
    if (tid == 0) data[n-1] = 0;
    /* down-sweep */
    for (int d = 1; d < n; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2*tid+1) - 1;
            int bi = offset * (2*tid+2) - 1;
            int t  = data[ai];
            data[ai] = data[bi];
            data[bi] += t;
        }
    }
    __syncthreads();
}

/* ══════════════════════════════════════════════════════════════════════════
 * TWO-PHASE PFAC KERNEL  (Lai et al. 2019, Figures 5, 7, 8)
 *
 * Phase 1 — Initial State Traversal (Figure 5):
 *   for start = 1 to THRESHOLD:
 *     if start < 2: use Prefix_PFAC_Table (shared memory)
 *     else: use Suffix_PFAC_Table (global memory)
 *     if TRAP_STATE → break
 *     if final state → record match
 *   if still active at threshold → mark Incomplete[tid] = 1
 *
 * Job Compression (Figure 7):
 *   prefix-sum on Incomplete → compact active threads to front of block
 *
 * Phase 2 — Remainder Traversal (Figure 8):
 *   Only remapped threads continue from saved (strIndex, next_state)
 * ══════════════════════════════════════════════════════════════════════════ */
__global__ void pfac_kernel(
    const unsigned char * __restrict__ text,
    long long                          text_len,
    const int           * __restrict__ suffix_table,  /* [num_states * ALPHA] global mem */
    const int           * __restrict__ prefix_table,  /* [ALPHA * ALPHA] — loaded to smem */
    const int           * __restrict__ is_final,      /* [num_states]                    */
    int                                num_states,
    int                                initial_state,
    unsigned long long  * __restrict__ match_count)
{
    /* shared memory layout:
       [0 .. BLOCK_SIZE-1]             : Incomplete[BlockSize]  (phase-1 flags)
       [BLOCK_SIZE .. 2*BLOCK_SIZE-1]  : prefixSum[BlockSize]   (scan buffer)
       [2*BLOCK_SIZE .. 2*BLOCK_SIZE+ALPHA*ALPHA-1] : Prefix PFAC Table (5×5)
       [2*BLOCK_SIZE+ALPHA*ALPHA .. ]: newStrIndex, newState arrays           */
    extern __shared__ int smem[];

    int *Incomplete    = smem;                          /* [BLOCK_SIZE]       */
    int *prefixSum     = smem + BLOCK_SIZE;             /* [BLOCK_SIZE]       */
    int *prefix_tbl_sm = smem + 2*BLOCK_SIZE;           /* [ALPHA*ALPHA]      */
    int *newStrIndex   = smem + 2*BLOCK_SIZE + ALPHA*ALPHA;  /* [BLOCK_SIZE]  */
    int *newState      = newStrIndex + BLOCK_SIZE;           /* [BLOCK_SIZE]  */

    int  ThreadID  = threadIdx.x;
    long long global_id = (long long)blockIdx.x * blockDim.x + ThreadID;

    /* load Prefix PFAC Table into shared memory (Figure 5 header) */
    if (ThreadID < ALPHA*ALPHA)
        prefix_tbl_sm[ThreadID] = prefix_table[ThreadID];
    __syncthreads();

    /* ── Phase 1: Initial State Traversal (Figure 5) ─────────────────────
     * strIndex starts at global_id (one thread per byte).
     * Loop from start=1 to THRESHOLD (inclusive).                          */
    int  next_state = initial_state;
    long long strIndex   = global_id;

    Incomplete[ThreadID] = 0;
    newStrIndex[ThreadID] = -1;
    newState[ThreadID]    = TRAP_STATE;

    if (global_id < text_len) {

        for (int start = 1; start <= THRESHOLD; start++) {
            long long idx = strIndex + start - 1;   /* 0-based position in text  */
            if (idx >= text_len) break;

            int a = dna_char(text[idx]);
            if (a < 0) { next_state = TRAP_STATE; break; }

            if (start < 2) {
                /* first character: use PREFIX table row from initial_state */
                /* prefix_tbl_sm[a1 * ALPHA + a2]: but start==1 means first char only
                   We use suffix table for char 1 from initial_state,
                   then prefix for chars 1-2 according to paper's 2-char prefix table. */
                /* Implementation: for start==1 look up col a from initial state in suffix */
                next_state = suffix_table[initial_state * ALPHA + a];
            } else if (start == 2) {
                /* second char: prefix_table[a1*ALPHA + a2] where a1 is char 0's alpha */
                /* Reconstruct: use suffix table from current next_state */
                next_state = suffix_table[next_state * ALPHA + a];
            } else {
                /* chars 3..THRESHOLD: Suffix PFAC Table */
                if (next_state == TRAP_STATE) break;
                next_state = suffix_table[next_state * ALPHA + a];
            }

            if (next_state == TRAP_STATE) break;

            /* check if final state (next_state < initial_state && next_state != 0) */
            if (is_final[next_state]) {
                atomicAdd(match_count, 1ULL);
                break;   /* PFAC: report at starting position, one match per thread */
            }
        }

        /* if still active at threshold → need Phase 2 */
        if (next_state != TRAP_STATE && !is_final[next_state]) {
            Incomplete[ThreadID] = 1;
            prefixSum[ThreadID]  = 1;
        } else {
            prefixSum[ThreadID] = 0;
        }
    } else {
        prefixSum[ThreadID] = 0;
    }
    __syncthreads();

    /* ── Job Compression (Figure 7): parallel prefix sum ─────────────────
     * Exclusive scan on prefixSum[] → gives target slot for each active thread */
    block_prefix_sum((volatile int *)prefixSum, ThreadID, BLOCK_SIZE);
    __syncthreads();

    /* active threads write their state to the compacted arrays */
    if (Incomplete[ThreadID] == 1) {
        int slot = prefixSum[ThreadID];   /* exclusive prefix sum = slot index */
        newStrIndex[slot] = (int)(global_id);   /* store thread id (add THRESHOLD later) */
        newState[slot]    = next_state;
    }
    __syncthreads();

    /* ── Phase 2: Remainder Traversal (Figure 8) ─────────────────────────
     * Only threads 0 .. k-1 (k = total incomplete) are active.
     * Each reads newStrIndex[ThreadID] and continues from newState[ThreadID]. */
    if (newStrIndex[ThreadID] != -1) {
        long long cont_start = (long long)newStrIndex[ThreadID] + THRESHOLD;
        int state2 = newState[ThreadID];

        while (cont_start < text_len && state2 != TRAP_STATE) {
            int a = dna_char(text[cont_start]);
            if (a < 0) { state2 = TRAP_STATE; break; }
            state2 = suffix_table[state2 * ALPHA + a];
            if (state2 == TRAP_STATE) break;
            if (is_final[state2]) {
                atomicAdd(match_count, 1ULL);
                break;
            }
            cont_start++;
        }
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 * CPU REFERENCE (for accuracy baseline & speedup)
 * Simple PFAC traversal on CPU — each starting position, no failure links.
 * ══════════════════════════════════════════════════════════════════════════ */
long long pfac_search_cpu(PFACAuto *pfac,
                           const char *text, long long len)
{
    long long matches = 0;
    for (long long i = 0; i < len; i++) {
        int state = pfac->initial_state;
        for (long long j = i; j < len; j++) {
            int a = dna_char((unsigned char)text[j]);
            if (a < 0) break;
            state = pfac->states[state].go[a];
            if (state == TRAP_STATE) break;
            if (pfac->states[state].is_final) { matches++; break; }
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
    char buf[MAX_PAT_LEN];
    int n = 0;
    while (fgets(buf, sizeof(buf), f) && n < MAX_PATTERNS) {
        int L = strlen(buf);
        while (L > 0 && (buf[L-1]=='\n'||buf[L-1]=='\r'||buf[L-1]==' ')) buf[--L]=0;
        if (L == 0 || buf[0] == '>') continue;
        pats[n++] = strdup(buf);
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
    long long sz = ftell(f); rewind(f);
    char *buf = (char *)malloc(sz + 1);
    fread(buf, 1, sz, f); buf[sz] = 0;
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

    /* ── load data ──────────────────────────────────────────────────────── */
    long long text_len;
    char *text = load_text(argv[1], &text_len);

    int    n_pat;
    char **patterns = load_patterns(argv[2], &n_pat);

    printf("=== Two-Phase PFAC DNA String Matching (CUDA) ===\n");
    printf("Text size   : %.2f MB (%lld bytes)\n", text_len/1e6, text_len);
    printf("Patterns    : %d\n\n", n_pat);

    /* ── build PFAC automaton ───────────────────────────────────────────── */
    PFACAuto pfac;
    pfac_build(&pfac, patterns, n_pat);
    printf("PFAC states  : %d\n", pfac.num_states);
    printf("Initial state: %d\n", pfac.initial_state);
    printf("Max pat len  : %d\n", pfac.max_pat_len);
    printf("Threshold    : %d (Phase-1 max transitions, per Tables 1&2 of paper)\n\n",
           THRESHOLD);

    /* ── build tables ───────────────────────────────────────────────────── */
    int *suffix_tbl  = pfac_flatten_suffix(&pfac);
    int *prefix_tbl  = pfac_build_prefix_table(&pfac);
    int *final_tbl   = pfac_final_table(&pfac);

    /* ── memory usage ───────────────────────────────────────────────────── */
    size_t suffix_bytes  = (size_t)pfac.num_states * ALPHA * sizeof(int);
    size_t prefix_bytes  = ALPHA * ALPHA * sizeof(int);
    size_t final_bytes   = (size_t)pfac.num_states * sizeof(int);
    size_t text_bytes    = (size_t)text_len;
    printf("[METRIC] Memory Usage (suffix table GPU): %.2f MB\n", suffix_bytes/1e6);
    printf("[METRIC] Memory Usage (prefix table smem): %.2f KB\n", prefix_bytes/1e3);
    printf("[METRIC] Memory Usage (text on GPU)      : %.2f MB\n\n", text_bytes/1e6);

    /* ── CPU reference ──────────────────────────────────────────────────── */
    timespec_t t0, t1; timer_init();
    clock_now(&t0);
    long long cpu_matches = pfac_search_cpu(&pfac, text, text_len);
    clock_now(&t1);
    double cpu_time_s = elapsed_sec(&t0, &t1);
    printf("[METRIC] CPU Execution Time : %.4f s\n", cpu_time_s);
    printf("[METRIC] CPU Matches        : %lld\n\n", cpu_matches);

    /* ── GPU setup ──────────────────────────────────────────────────────── */
    unsigned char *d_text;
    int *d_suffix, *d_prefix, *d_final;
    unsigned long long *d_matches, h_matches = 0;

    check_cuda(cudaMalloc(&d_text,    text_bytes),   "malloc text");
    check_cuda(cudaMalloc(&d_suffix,  suffix_bytes),  "malloc suffix");
    check_cuda(cudaMalloc(&d_prefix,  prefix_bytes),  "malloc prefix");
    check_cuda(cudaMalloc(&d_final,   final_bytes),   "malloc final");
    check_cuda(cudaMalloc(&d_matches, sizeof(unsigned long long)), "malloc cnt");

    check_cuda(cudaMemcpy(d_text,   text,        text_bytes,   cudaMemcpyHostToDevice), "cp text");
    check_cuda(cudaMemcpy(d_suffix, suffix_tbl,  suffix_bytes, cudaMemcpyHostToDevice), "cp suffix");
    check_cuda(cudaMemcpy(d_prefix, prefix_tbl,  prefix_bytes, cudaMemcpyHostToDevice), "cp prefix");
    check_cuda(cudaMemcpy(d_final,  final_tbl,   final_bytes,  cudaMemcpyHostToDevice), "cp final");
    check_cuda(cudaMemcpy(d_matches, &h_matches, sizeof(unsigned long long),
                          cudaMemcpyHostToDevice), "cp cnt");

    /* ── kernel launch ──────────────────────────────────────────────────── */
    /* one thread per byte of input (PFAC principle: Lai et al. Section 1) */
    long long n_threads = text_len;
    int  n_blocks  = (int)((n_threads + BLOCK_SIZE - 1) / BLOCK_SIZE);

    /* shared memory: Incomplete + prefixSum + prefix_tbl + newStrIndex + newState */
    size_t smem_bytes = (2 * BLOCK_SIZE + ALPHA*ALPHA + 2*BLOCK_SIZE) * sizeof(int);

    /* ── theoretical occupancy ──────────────────────────────────────────── */
    int max_active_blocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks, pfac_kernel, BLOCK_SIZE, smem_bytes);
    int device_id; cudaGetDevice(&device_id);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, device_id);
    float occupancy = (max_active_blocks * BLOCK_SIZE) /
                      (float)prop.maxThreadsPerMultiProcessor;
    printf("[METRIC] Theoretical Occupancy : %.2f %%\n", occupancy * 100.0f);
    printf("[METRIC] SM count              : %d\n", prop.multiProcessorCount);
    printf("[METRIC] Shared mem / block    : %.2f KB\n\n", smem_bytes/1e3);

    /* ── kernel timing ──────────────────────────────────────────────────── */
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    cudaEventRecord(ev_start);
    pfac_kernel<<<n_blocks, BLOCK_SIZE, smem_bytes>>>(
        d_text, text_len,
        d_suffix, d_prefix, d_final,
        pfac.num_states, pfac.initial_state,
        d_matches);
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    check_cuda(cudaGetLastError(), "pfac kernel");

    float kernel_ms;
    cudaEventElapsedTime(&kernel_ms, ev_start, ev_stop);
    double gpu_time_s = kernel_ms / 1000.0;

    /* ── retrieve results ───────────────────────────────────────────────── */
    check_cuda(cudaMemcpy(&h_matches, d_matches, sizeof(unsigned long long),
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
    printf("[METRIC] SpeedUp (GPU vs CPU)    : %.2f x\n",    speedup);
    printf("[METRIC] GPU Matches             : %lld\n",       h_matches);
    printf("[METRIC] CPU Matches (reference) : %lld\n",       cpu_matches);
    printf("[METRIC] Accuracy                : %.2f %%\n",    accuracy);
    printf("\n[NOTE] For Warp Efficiency, Branch Divergence, Memory Throughput:\n");
    printf("       nsys profile --stats=true ./pfac_dna <text> <patterns>\n");
    printf("       nv-nsight-cu-cli --metrics \\\n");
    printf("         smsp__sass_average_branch_targets_threads_uniform.pct,\\\n");
    printf("         smsp__warps_active.avg.pct_of_peak_sustained_active,\\\n");
    printf("         l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum \\\n");
    printf("         ./pfac_dna <text> <patterns>\n\n");

    printf("[METRIC] Scalability: run with VDB → CHM13_rna → CHM13_genomic\n");
    printf("         → T.aestivum and record Kernel Time + Throughput per dataset.\n\n");

    /* ── cleanup ────────────────────────────────────────────────────────── */
    cudaFree(d_text); cudaFree(d_suffix); cudaFree(d_prefix);
    cudaFree(d_final); cudaFree(d_matches);
    cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
    free(suffix_tbl); free(prefix_tbl); free(final_tbl);
    free(pfac.states); free(text);
    for (int i = 0; i < n_pat; i++) free(patterns[i]);
    free(patterns);

    return 0;
}

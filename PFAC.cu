/**
 * PFAC.cu  —  Parallel Failure-less Aho-Corasick (GPU)
 * Reference: Gagniuc et al., 2025 — Algorithms 18(12), 742.
 *
 * Perbaikan: Menambahkan Sequence Generator sesuai Research Guide.
 * * Compile: nvcc -O2 -o PFAC PFAC.cu
 * Run Real:    ./PFAC.exe raw.txt
 * Run Benchmark: ./PFAC.exe dummy.txt 65536
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

#define ALPHA_SIZE         4
#define MAX_STATES      5000
#define MAX_PATTERNS      64
#define THREADS_PER_BLOCK 256

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[CUDA ERROR] %s at %s:%d\n", \
            cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); } } while(0)

// Generator DNA Random sesuai Research Guide [cite: 76, 78]
void generate_random_dna(char *seq, long long n) {
    const char bases[] = "ACGT";
    srand((unsigned int)time(NULL));
    for (long long i = 0; i < n; i++) {
        seq[i] = bases[rand() % 4]; // [cite: 80, 81]
    }
    seq[n] = '\0';
}

__device__ __forceinline__ int d_dna_map(char c) {
    if (c=='A'||c=='a') return 0;
    if (c=='C'||c=='c') return 1;
    if (c=='G'||c=='g') return 2;
    if (c=='T'||c=='t') return 3;
    return -1;
}

static int h_dna_map(char c) {
    switch(c) {
        case 'A': case 'a': return 0; case 'C': case 'c': return 1;
        case 'G': case 'g': return 2; case 'T': case 't': return 3;
        default: return -1;
    }
}

typedef struct {
    int delta[MAX_STATES][ALPHA_SIZE];
    unsigned long long O[MAX_STATES];
    int numStates;
} PFACAutomaton;

static void insertPattern(PFACAutomaton *ac, const char *pat, int idx) {
    int state = 0; // Mulai dari ROOT [cite: 314]
    for (int i = 0; pat[i]; i++) {
        int c = h_dna_map(pat[i]);
        if (c == -1) continue;
        if (ac->delta[state][c] == -1)
            ac->delta[state][c] = ac->numStates++; // Membangun "Tree"/Trie [cite: 313]
        state = ac->delta[state][c];
    }
    ac->O[state] |= (1ULL << idx); // Terminal state [cite: 318]
}

static void buildFailurelessTable(PFACAutomaton *ac) {
    int queue[MAX_STATES], head = 0, tail = 0;
    int f[MAX_STATES]; memset(f, 0, sizeof(f));
    for (int c = 0; c < ALPHA_SIZE; c++)
        if (ac->delta[0][c] != -1) queue[tail++] = ac->delta[0][c];
    while (head < tail) {
        int v = queue[head++];
        for (int c = 0; c < ALPHA_SIZE; c++) {
            int u = ac->delta[v][c];
            if (u == -1) { ac->delta[v][c] = ac->delta[f[v]][c]; }
            else {
                f[u] = ac->delta[f[v]][c];
                if (f[u] == -1) f[u] = 0;
                ac->O[u] |= ac->O[f[u]];
                queue[tail++] = u;
            }
        }
    }
}

__global__ void pfacKernel(
    const char *text, int n, const int *delta,
    const unsigned long long *output,
    int *matchCount, unsigned long long *motifCounts, int numMotifs
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // Thread-per-position [cite: 226, 500]
    if (tid >= n) return;
    int state = 0;
    for (int i = tid; i < n; i++) {
        int c = d_dna_map(text[i]);
        if (c == -1) break;
        int next = delta[state * ALPHA_SIZE + c];
        if (next == -1) break; // Failure-less: stop jika mismatch [cite: 207, 226]
        state = next;
        if (output[state] != 0) {
            atomicAdd(matchCount, 1);
            for (int p = 0; p < numMotifs; p++)
                if (output[state] & (1ULL << p))
                    atomicAdd((unsigned long long *)&motifCounts[p], 1ULL);
            break;
        }
    }
}

static const char *MOTIF_NAMES[] = {
    "HOMOPOLYMER_A4","HOMOPOLYMER_C4","HOMOPOLYMER_G4","HOMOPOLYMER_T4",
    "ECORI_GAATTC","BAMHI_GGATCC","HINDIII_AAGCTT","XHOI_CTCGAG",
    "SALI_GTCGAC","NCOI_CCATGG","NDEI_CATATG","SPHI_GCATGC",
    "REPEAT_ATATAT","REPEAT_TATATA","REPEAT_CGCGCG","REPEAT_GCGCGC",
    "STR_AAGAAG","STR_CAGCAG","STR_TGCTGC",
    "GC_RUN_GCGCGCGC","AT_RUN_ATATATATAT",
};
static const char *MOTIF_SEQS[] = {
    "AAAA","CCCC","GGGG","TTTT",
    "GAATTC","GGATCC","AAGCTT","CTCGAG",
    "GTCGAC","CCATGG","CATATG","GCATGC",
    "ATATAT","TATATA","CGCGCG","GCGCGC",
    "AAGAAG","CAGCAG","TGCTGC",
    "GCGCGCGC","ATATATATAT",
};
static const int NUM_MOTIFS = 21;

int main(int argc, char *argv[]) {
    const char *input_file = "raw.txt";
    long long use_n = -1;
    if (argc > 1) input_file = argv[1];
    if (argc > 2) use_n = atoll(argv[2]);

    printf("========================================================\n");
    printf("  Parallel Failure-less AC (PFAC) -- GPU Kernel\n");
    printf("  Gagniuc et al., Algorithms 2025, 18, 742\n");
    printf("========================================================\n\n");

    PFACAutomaton *ac = (PFACAutomaton *)malloc(sizeof(PFACAutomaton));
    memset(ac->delta, -1, sizeof(ac->delta));
    memset(ac->O, 0, sizeof(ac->O));
    ac->numStates = 1;

    printf("[PFAC-Build] Loading %d motifs...\n", NUM_MOTIFS);
    for (int i = 0; i < NUM_MOTIFS; i++) insertPattern(ac, MOTIF_SEQS[i], i);
    buildFailurelessTable(ac);
    printf("[PFAC-Build] States: %d\n\n", ac->numStates);

    char *h_text;
    long long n;

    FILE *fp = fopen(input_file, "r");
    if (!fp) {
        if (use_n > 0) {
            printf("[INFO] File %s not found. Generating %lld random bases (Research Guide Mode)...\n", input_file, use_n);
            n = use_n;
            h_text = (char *)malloc(n + 1);
            generate_random_dna(h_text, n); // [cite: 75, 78]
        } else {
            printf("[ERROR] Cannot open %s and no size N provided.\n", input_file);
            return 1;
        }
    } else {
        fseek(fp, 0, SEEK_END);
        long long file_size = ftell(fp); fseek(fp, 0, SEEK_SET);
        n = (use_n > 0 && use_n < file_size) ? use_n : file_size;
        // Cap 1GB untuk stabilitas memori GPU
        if (n > 1073741824LL) n = 1073741824LL; 
        h_text = (char *)malloc(n + 1);
        fread(h_text, 1, n, fp);
        fclose(fp); h_text[n] = '\0';
    }

    printf("[PFAC] Scanning %lld bases on GPU...\n\n", n);

    int *h_flat = (int *)malloc((size_t)ac->numStates * ALPHA_SIZE * sizeof(int));
    for (int s = 0; s < ac->numStates; s++)
        for (int c = 0; c < ALPHA_SIZE; c++)
            h_flat[s * ALPHA_SIZE + c] = ac->delta[s][c];

    char *d_text; int *d_delta; unsigned long long *d_output, *d_motif;
    int *d_match, h_match = 0;
    unsigned long long h_motif[MAX_PATTERNS] = {0};
    size_t dsz = (size_t)ac->numStates*ALPHA_SIZE*sizeof(int);
    size_t osz = (size_t)ac->numStates*sizeof(unsigned long long);
    size_t msz = (size_t)MAX_PATTERNS*sizeof(unsigned long long);

    CUDA_CHECK(cudaMalloc(&d_text, n));
    CUDA_CHECK(cudaMalloc(&d_delta, dsz));
    CUDA_CHECK(cudaMalloc(&d_output, osz));
    CUDA_CHECK(cudaMalloc(&d_match, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_motif, msz));
    CUDA_CHECK(cudaMemcpy(d_text, h_text, n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_delta, h_flat, dsz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output, ac->O, osz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_match, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_motif, 0, msz));

    cudaEvent_t s_event, e_event; 
    cudaEventCreate(&s_event); cudaEventCreate(&e_event);
    cudaEventRecord(s_event);
    
    int blocks = (int)((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    pfacKernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_text, (int)n, d_delta, d_output, d_match, d_motif, NUM_MOTIFS);
    
    cudaEventRecord(e_event); cudaEventSynchronize(e_event);
    float ms = 0; cudaEventElapsedTime(&ms, s_event, e_event);

    CUDA_CHECK(cudaMemcpy(&h_match, d_match, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_motif, d_motif, msz, cudaMemcpyDeviceToHost));

    printf("[Results]\n");
    printf("  Total matches : %d\n", h_match);
    printf("  Kernel time   : %.4f ms\n", ms);
    printf("  Throughput    : %.2f GB/s\n\n", (n/1e9)/(ms/1000.0));
    printf("  %-28s  Count\n", "Motif");
    printf("  %-28s  -----\n", "----------------------------");
    for (int p = 0; p < NUM_MOTIFS; p++)
        if (h_motif[p] > 0)
            printf("  %-28s  %llu\n", MOTIF_NAMES[p], h_motif[p]);

    cudaEventDestroy(s_event); cudaEventDestroy(e_event);
    cudaFree(d_text); cudaFree(d_delta); cudaFree(d_output);
    cudaFree(d_match); cudaFree(d_motif);
    free(ac); free(h_text); free(h_flat);
    printf("\n[Done] PFAC GPU scan complete.\n");
    return 0;
}
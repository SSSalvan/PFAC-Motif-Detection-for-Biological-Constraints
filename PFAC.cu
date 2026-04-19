/**
 * PFAC.cu
 *
 * Parallel Failureless Aho-Corasick (PFAC) implementation for DNA Motif Detection.
 * Optimized for GPU-based high-throughput scanning of biological sequences.
 * Part of: Constraint-Aware PFAC Motif Matching Research.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define ALPHA_SIZE    4     // DNA: A, C, G, T
#define MAX_STATES    20000
#define THREADS_PER_BLOCK 256
#define MAX_INPUT_LEN 10000000 // 10 MB

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// DNA Mapping (Device)
__device__ int d_dna_map(char c) {
    if (c == 'A' || c == 'a') return 0;
    if (c == 'C' || c == 'c') return 1;
    if (c == 'G' || c == 'g') return 2;
    if (c == 'T' || c == 't') return 3;
    return -1;
}

// Host-side mapping
int h_dna_map(char c) {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default: return -1;
    }
}

typedef struct {
    int delta[MAX_STATES][ALPHA_SIZE];
    unsigned long long O[MAX_STATES];
    int numStates;
} AhoCorasick;

// CPU: Build the failureless automaton (all transitions lead to a state or back to root)
void insertPattern(AhoCorasick *ac, const char *pattern, int patIndex) {
    int state = 0;
    for (int i = 0; pattern[i] != '\0'; i++) {
        int c = h_dna_map(pattern[i]);
        if (c == -1) continue;
        if (ac->delta[state][c] == -1) {
            ac->delta[state][c] = ac->numStates++;
        }
        state = ac->delta[state][c];
    }
    ac->O[state] |= (1ULL << patIndex);
}

// In PFAC, we pre-bake failure paths into the transition table
void buildFailurelessTable(AhoCorasick *ac) {
    int queue[MAX_STATES];
    int head = 0, tail = 0;
    int f[MAX_STATES] = {0};

    for (int c = 0; c < ALPHA_SIZE; c++) {
        if (ac->delta[0][c] != -1) {
            queue[tail++] = ac->delta[0][c];
        } else {
            ac->delta[0][c] = 0;
        }
    }

    while (head < tail) {
        int v = queue[head++];
        for (int c = 0; c < ALPHA_SIZE; c++) {
            int u = ac->delta[v][c];
            if (u == -1) {
                ac->delta[v][c] = ac->delta[f[v]][c];
            } else {
                f[u] = ac->delta[f[v]][c];
                ac->O[u] |= ac->O[f[u]];
                queue[tail++] = u;
            }
        }
    }
}

// GPU Kernel: One thread per starting position
__global__ void pfacKernel(const char *text, int n, const int *delta, const unsigned long long *output, int *matchCount, int numStates) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int state = 0;
    // Scan forward from tid
    for (int i = tid; i < n; i++) {
        int c = d_dna_map(text[i]);
        if (c == -1) break; // Unknown character
        
        state = delta[state * ALPHA_SIZE + c];
        
        if (output[state] != 0) {
            atomicAdd(matchCount, 1);
            // In some PFAC versions, we stop at first match from this position
            break; 
        }
        
        if (state == 0) break; // Reached root, no match starting here
    }
}

int main(int argc, char *argv[]) {
    const char *input_file = "human_raw.txt";
    if (argc > 1) input_file = argv[1];

    const char *motifs[] = {
        "AAAAAA", "CCCCCC", "GGGGGG", "TTTTTT", 
        "CGCGCG", "ATATAT"
    };
    int numMotifs = 6;

    AhoCorasick *ac = (AhoCorasick *)malloc(sizeof(AhoCorasick));
    memset(ac->delta, -1, sizeof(ac->delta));
    memset(ac->O, 0, sizeof(ac->O));
    ac->numStates = 1;

    for (int i = 0; i < numMotifs; i++) {
        insertPattern(ac, motifs[i], i);
    }
    buildFailurelessTable(ac);

    // Read DNA sequence from file
    FILE *f = fopen(input_file, "r");
    if (!f) {
        printf("Error: Could not open %s. Run dna_loader.py first.\n", input_file);
        return 1;
    }
    char *h_text = (char *)malloc(MAX_INPUT_LEN);
    int n = fread(h_text, 1, MAX_INPUT_LEN, f);
    fclose(f);

    printf("Scanning DNA sequence (%d bases) with Parallel Failureless AC (GPU)...\n", n);

    // Prepare GPU data
    char *d_text;
    int *d_delta;
    unsigned long long *d_output;
    int *d_matchCount;
    int h_matchCount = 0;

    int *h_flat_delta = (int *)malloc(ac->numStates * ALPHA_SIZE * sizeof(int));
    for(int s=0; s<ac->numStates; s++) {
        for(int c=0; c<ALPHA_SIZE; c++) {
            h_flat_delta[s * ALPHA_SIZE + c] = ac->delta[s][c];
        }
    }

    CUDA_CHECK(cudaMalloc(&d_text, n));
    CUDA_CHECK(cudaMalloc(&d_delta, ac->numStates * ALPHA_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output, ac->numStates * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_matchCount, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_text, h_text, n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_delta, h_flat_delta, ac->numStates * ALPHA_SIZE * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output, ac->O, ac->numStates * sizeof(unsigned long long), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_matchCount, 0, sizeof(int)));

    // Timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    pfacKernel<<<blocks, THREADS_PER_BLOCK>>>(d_text, n, d_delta, d_output, d_matchCount, ac->numStates);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    CUDA_CHECK(cudaMemcpy(&h_matchCount, d_matchCount, sizeof(int), cudaMemcpyDeviceToHost));

    printf("Total matches found: %d\n", h_matchCount);
    printf("Kernel execution time: %.4f ms\n", milliseconds);

    // Cleanup
    cudaFree(d_text);
    cudaFree(d_delta);
    cudaFree(d_output);
    cudaFree(d_matchCount);
    free(ac);
    free(h_text);
    free(h_flat_delta);

    return 0;
}

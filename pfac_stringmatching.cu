#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define ALPHA_SIZE 4
#define MAX_STATES 2048
#define THREADS_PER_BLOCK 256

// Macro for catching CUDA errors
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// DNA Mapping: A=0, C=1, G=2, T=3
__device__ int get_char_idx(char c) {
    if (c == 'A' || c == 'a') return 0;
    if (c == 'C' || c == 'c') return 1;
    if (c == 'G' || c == 'g') return 2;
    if (c == 'T' || c == 't') return 3;
    return -1;
}

__global__ void pfac_kernel(const char* text, int text_len, const int* dfa, const int* out, int* count) {
    // Shared memory optimization (PFAC-LC)
    __shared__ int s_dfa[MAX_STATES * ALPHA_SIZE];
    __shared__ int s_out[MAX_STATES];

    int tid = threadIdx.x;
    
    // Cooperative load into Shared Memory[cite: 2]
    for (int i = tid; i < MAX_STATES * ALPHA_SIZE; i += blockDim.x) {
        s_dfa[i] = dfa[i];
    }
    for (int i = tid; i < MAX_STATES; i += blockDim.x) {
        s_out[i] = out[i];
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < text_len) {
        int state = 0;
        // Each thread scans from its assigned character[cite: 2]
        for (int j = idx; j < text_len; j++) {
            int c = get_char_idx(text[j]);
            if (c == -1) break; // Invalid character = Thread annihilation[cite: 2]

            state = s_dfa[state * ALPHA_SIZE + c];
            if (state == -1) break; // No transition = Thread annihilation[cite: 2]

            if (s_out[state] > 0) {
                atomicAdd(count, 1); // Match found![cite: 2]
            }
        }
    }
}

// Host-side Goto Function construction[cite: 2]
void build_pfac_dfa(int* h_dfa, int* h_out, const char** patterns, int num_pats) {
    int next_free_state = 1;
    for (int i = 0; i < num_pats; i++) {
        int curr = 0;
        const char* p = patterns[i];
        for (int j = 0; p[j] != '\0'; j++) {
            int c = (p[j] == 'A') ? 0 : (p[j] == 'C') ? 1 : (p[j] == 'G') ? 2 : 3;
            if (h_dfa[curr * ALPHA_SIZE + c] == -1) {
                h_dfa[curr * ALPHA_SIZE + c] = next_free_state++;
            }
            curr = h_dfa[curr * ALPHA_SIZE + c];
        }
        h_out[curr] = 1; // Mark this state as an output/match state[cite: 2]
    }
}

int main() {
    // 1. Data Setup
    const char* h_text = "ACGTACGTACGT"; // 3 matches for ACGT, 2 for CGTA
    const char* h_patterns[] = {"ACGT", "CGTA"};
    int text_len = strlen(h_text);
    int num_pats = 2;

    // 2. Allocate Host Memory
    int* h_dfa = (int*)malloc(MAX_STATES * ALPHA_SIZE * sizeof(int));
    int* h_out = (int*)malloc(MAX_STATES * sizeof(int));
    memset(h_dfa, -1, MAX_STATES * ALPHA_SIZE * sizeof(int));
    memset(h_out, 0, MAX_STATES * sizeof(int));

    // 3. Build DFA logic
    build_pfac_dfa(h_dfa, h_out, h_patterns, num_pats);

    // 4. Allocate Device Memory
    char* d_text;
    int *d_dfa, *d_out, *d_count;
    int h_count = 0;

    cudaError_t init_err = cudaFree(0); 
    if (init_err != cudaSuccess) {
        printf("CUDA Init Error: %s\n", cudaGetErrorString(init_err));
        return 1;
    }
    
    gpuErrchk(cudaMalloc(&d_text, text_len));
    gpuErrchk(cudaMalloc(&d_dfa, MAX_STATES * ALPHA_SIZE * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_out, MAX_STATES * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_count, sizeof(int)));

    // 5. Transfer Data to GPU[cite: 2]
    gpuErrchk(cudaMemcpy(d_text, h_text, text_len, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_dfa, h_dfa, MAX_STATES * ALPHA_SIZE * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_out, h_out, MAX_STATES * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_count, &h_count, sizeof(int), cudaMemcpyHostToDevice));

    // 6. Launch Kernel
    int blocks = (text_len + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    printf("Launching %d blocks on GPU...\n", blocks);
    pfac_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_text, text_len, d_dfa, d_out, d_count);
    
    // Check for launch errors
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // 7. Get Results
    gpuErrchk(cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));

    printf("----------------------------------\n");
    printf("Total Patterns Found: %d\n", h_count);
    printf("----------------------------------\n");

    // Cleanup
    cudaFree(d_text); cudaFree(d_dfa); cudaFree(d_out); cudaFree(d_count);
    free(h_dfa); free(h_out);

    return 0;
}
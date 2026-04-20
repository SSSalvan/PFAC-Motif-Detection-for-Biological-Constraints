/**
 * AC_Sequential.cu
 *
 * Standard Sequential Aho-Corasick implementation for DNA Motif Detection.
 * Uses classic failure links and a single-threaded CPU evaluation.
 * Used as a baseline to compare against GPU-based PFAC.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h> // Included so it cleanly compiles with nvcc

#define ALPHA_SIZE    4      // DNA: A, C, G, T
#define MAX_STATES    20000
#define MAX_INPUT_LEN 3221225472LL // Update this based on your raw.txt size

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
    int fail[MAX_STATES];  // ADDED: Standard AC failure links
    unsigned long long O[MAX_STATES];
    int numStates;
} AhoCorasick;

// CPU: Build the basic Trie structure
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

// CPU: Build the Failure Table (Standard Aho-Corasick logic)
void buildFailureTable(AhoCorasick *ac) {
    int queue[MAX_STATES];
    int head = 0, tail = 0;

    ac->fail[0] = 0;

    // 1. Set failure links for depth-1 nodes to root (0)
    for (int c = 0; c < ALPHA_SIZE; c++) {
        int u = ac->delta[0][c];
        if (u != -1) {
            ac->fail[u] = 0;
            queue[tail++] = u;
        } else {
            // Root transitions to itself for missing characters
            ac->delta[0][c] = 0; 
        }
    }

    // 2. BFS to set failure links for the rest of the Trie
    while (head < tail) {
        int r = queue[head++];
        for (int c = 0; c < ALPHA_SIZE; c++) {
            int u = ac->delta[r][c];
            if (u != -1) {
                queue[tail++] = u;
                
                // Follow the failure link of the parent
                int v = ac->fail[r];
                while (ac->delta[v][c] == -1) {
                    v = ac->fail[v];
                }
                
                ac->fail[u] = ac->delta[v][c];
                
                // Merge output flags so we catch sub-patterns
                ac->O[u] |= ac->O[ac->fail[u]];
            }
        }
    }
}

// CPU: Sequential Standard AC Search
int searchSequentialAC(const char *text, int n, AhoCorasick *ac) {
    int matchCount = 0;
    int state = 0;

    for (int i = 0; i < n; i++) {
        int c = h_dna_map(text[i]);
        
        // If unknown character (like 'N' in DNA), reset state machine
        if (c == -1) {
            state = 0;
            continue;
        }

        // Follow failure links until we find a valid transition
        while (ac->delta[state][c] == -1) {
            state = ac->fail[state];
        }
        
        // Take the transition
        state = ac->delta[state][c];

        // If this state has an output flag, we found a match
        if (ac->O[state] != 0) {
            matchCount++;
        }
    }

    return matchCount;
}

int main(int argc, char *argv[]) {
    const char *input_file = "raw.txt";
    if (argc > 1) input_file = argv[1];

    const char *motifs[] = {
        "AAAAAA", "CCCCCC", "GGGGGG", "TTTTTT", 
        "CGCGCG", "ATATAT"
    };
    int numMotifs = 6;

    AhoCorasick *ac = (AhoCorasick *)malloc(sizeof(AhoCorasick));
    memset(ac->delta, -1, sizeof(ac->delta));
    memset(ac->fail, 0, sizeof(ac->fail));
    memset(ac->O, 0, sizeof(ac->O));
    ac->numStates = 1;

    // 1. Build Trie
    for (int i = 0; i < numMotifs; i++) {
        insertPattern(ac, motifs[i], i);
    }
    
    // 2. Build Failure Links (Crucial for Standard AC)
    buildFailureTable(ac);

    // 3. Read DNA sequence from file
    FILE *f = fopen(input_file, "r");
    if (!f) {
        printf("Error: Could not open %s.\n", input_file);
        return 1;
    }
    
    char *h_text = (char *)malloc(MAX_INPUT_LEN);
    if (!h_text) {
        printf("Error: Failed to allocate memory for DNA sequence.\n");
        fclose(f);
        return 1;
    }
    
    int n = fread(h_text, 1, MAX_INPUT_LEN, f);
    fclose(f);

    printf("Scanning DNA sequence (%d bases) with Sequential Standard AC (CPU)...\n", n);

    // 4. Time the Sequential CPU Search using CUDA Events (for apples-to-apples comparison)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int h_matchCount = searchSequentialAC(h_text, n, ac);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Total matches found: %d\n", h_matchCount);
    printf("CPU Execution time: %.4f ms\n", milliseconds);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(ac);
    free(h_text);

    return 0;
}
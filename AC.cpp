/**
 * AC.cu
 *
 * Sequential Aho-Corasick implementation for DNA Motif Detection.
 * Optimized for CPU-based sequential scanning of biological sequences.
 * Part of: Constraint-Aware PFAC Motif Matching Research.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ALPHA_SIZE    4     // DNA: A, C, G, T
#define MAX_STATES    20000
#define MAX_PATTERNS  64
#define MAX_INPUT_LEN 3000000000ULL // 3 GB

// DNA Mapping: A=0, C=1, G=2, T=3
int dna_map(char c) {
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
    int f[MAX_STATES];
    unsigned long long O[MAX_STATES];
    int numStates;
} AhoCorasick;

void insertPattern(AhoCorasick *ac, const char *pattern, int patIndex) {
    int state = 0;
    for (int i = 0; pattern[i] != '\0'; i++) {
        int c = dna_map(pattern[i]);
        if (c == -1) continue;
        if (ac->delta[state][c] == -1) {
            ac->delta[ac->numStates][0] = -1; // init next state
            ac->delta[state][c] = ac->numStates++;
        }
        state = ac->delta[state][c];
    }
    ac->O[state] |= (1ULL << patIndex);
}

void buildFailureLinks(AhoCorasick *ac) {
    int queue[MAX_STATES];
    int head = 0, tail = 0;

    for (int c = 0; c < ALPHA_SIZE; c++) {
        if (ac->delta[0][c] != -1) {
            ac->f[ac->delta[0][c]] = 0;
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
                ac->delta[v][c] = ac->delta[ac->f[v]][c];
            } else {
                ac->f[u] = ac->delta[ac->f[v]][c];
                ac->O[u] |= ac->O[ac->f[u]];
                queue[tail++] = u;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    const char *input_file = "raw.txt";
    if (argc > 1) input_file = argv[1];

    // Load biological constraints
    const char *motifs[] = {
        "AAAAAA", "CCCCCC", "GGGGGG", "TTTTTT", 
        "CGCGCG", "ATATAT"
    };
    int numMotifs = 6;

    AhoCorasick *ac = (AhoCorasick *)malloc(sizeof(AhoCorasick));
    memset(ac->delta, -1, sizeof(ac->delta));
    memset(ac->f, 0, sizeof(ac->f));
    memset(ac->O, 0, sizeof(ac->O));
    ac->numStates = 1;

    for (int i = 0; i < numMotifs; i++) {
        insertPattern(ac, motifs[i], i);
    }
    buildFailureLinks(ac);

    // Read DNA sequence from file
    FILE *f = fopen(input_file, "r");
    if (!f) {
        printf("Error: Could not open %s. Run dna_loader.py first.\n", input_file);
        return 1;
    }
    char *text = (char *)malloc(MAX_INPUT_LEN);
    int n = fread(text, 1, MAX_INPUT_LEN, f);
    fclose(f);

    printf("Scanning DNA sequence (%d bases) with Sequential AC...\n", n);
    
    clock_t start = clock();
    int state = 0;
    int matches = 0;
    for (int i = 0; i < n; i++) {
        int c = dna_map(text[i]);
        if (c == -1) continue;
        state = ac->delta[state][c];
        if (ac->O[state] != 0) {
            matches++;
        }
    }
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Total matches found: %d\n", matches);
    printf("Time spent: %.4f seconds\n", time_spent);

    free(ac);
    free(text);
    return 0;
}

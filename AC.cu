/**
 * AC.cu  —  Sequential Aho-Corasick (CPU Baseline)
 * Reference: Gagniuc et al., 2025 — Algorithms 18(12), 742.
 *
 * Compile: nvcc -O2 -o AC AC.cu
 * Run:     ./AC.exe raw.txt
 * Run N:   ./AC.exe raw.txt 65536
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define ALPHA_SIZE    4
#define MAX_STATES    20000
#define MAX_PATTERNS  64

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

static int h_dna_map(char c) {
    switch(c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default: return -1;
    }
}

typedef struct {
    int delta[MAX_STATES][ALPHA_SIZE];
    int fail[MAX_STATES];
    unsigned long long O[MAX_STATES];
    int numStates;
} AhoCorasick;

static void insertPattern(AhoCorasick *ac, const char *pat, int idx) {
    int state = 0;
    for (int i = 0; pat[i]; i++) {
        int c = h_dna_map(pat[i]);
        if (c == -1) continue;
        if (ac->delta[state][c] == -1)
            ac->delta[state][c] = ac->numStates++;
        state = ac->delta[state][c];
    }
    ac->O[state] |= (1ULL << idx);
}

static void buildFailureTable(AhoCorasick *ac) {
    int queue[MAX_STATES], head = 0, tail = 0;
    ac->fail[0] = 0;
    for (int c = 0; c < ALPHA_SIZE; c++) {
        int u = ac->delta[0][c];
        if (u != -1) { ac->fail[u] = 0; queue[tail++] = u; }
        else ac->delta[0][c] = 0;
    }
    while (head < tail) {
        int r = queue[head++];
        for (int c = 0; c < ALPHA_SIZE; c++) {
            int u = ac->delta[r][c];
            if (u != -1) {
                queue[tail++] = u;
                int v = ac->fail[r];
                while (ac->delta[v][c] == -1) v = ac->fail[v];
                ac->fail[u] = ac->delta[v][c];
                ac->O[u] |= ac->O[ac->fail[u]];
            }
        }
    }
}

static int searchAC(const char *text, long long n, AhoCorasick *ac,
                    unsigned long long *motifCounts) {
    int matches = 0, state = 0;
    for (long long i = 0; i < n; i++) {
        int c = h_dna_map(text[i]);
        if (c == -1) { state = 0; continue; }
        while (ac->delta[state][c] == -1) state = ac->fail[state];
        state = ac->delta[state][c];
        if (ac->O[state]) {
            matches++;
            for (int p = 0; p < NUM_MOTIFS; p++)
                if (ac->O[state] & (1ULL << p)) motifCounts[p]++;
        }
    }
    return matches;
}

int main(int argc, char *argv[]) {
    const char *input_file = "raw.txt";
    long long use_n = -1;
    if (argc > 1) input_file = argv[1];
    if (argc > 2) use_n = atoll(argv[2]);

    printf("========================================================\n");
    printf("  Sequential Aho-Corasick (AC) -- CPU Baseline\n");
    printf("  Gagniuc et al., Algorithms 2025, 18, 742\n");
    printf("========================================================\n\n");

    AhoCorasick *ac = (AhoCorasick *)malloc(sizeof(AhoCorasick));
    memset(ac->delta, -1, sizeof(ac->delta));
    memset(ac->fail,   0, sizeof(ac->fail));
    memset(ac->O,      0, sizeof(ac->O));
    ac->numStates = 1;

    printf("[AC-Build] Loading %d motifs...\n", NUM_MOTIFS);
    for (int i = 0; i < NUM_MOTIFS; i++) insertPattern(ac, MOTIF_SEQS[i], i);
    buildFailureTable(ac);
    printf("[AC-Build] States: %d\n\n", ac->numStates);

    FILE *fp = fopen(input_file, "r");
    if (!fp) { printf("[ERROR] Cannot open %s\n", input_file); return 1; }
    fseek(fp, 0, SEEK_END);
    long long file_size = ftell(fp); fseek(fp, 0, SEEK_SET);
    long long alloc_n = (use_n > 0 && use_n < file_size) ? use_n : file_size;
    char *h_text = (char *)malloc(alloc_n + 1);
    long long n = (long long)fread(h_text, 1, alloc_n, fp);
    fclose(fp); h_text[n] = '\0';

    printf("[AC] Scanning %lld bases...\n\n", n);
    unsigned long long motifCounts[MAX_PATTERNS] = {0};

    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    int matches = searchAC(h_text, n, ac, motifCounts);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e);

    printf("[Results]\n");
    printf("  Total matches : %d\n", matches);
    printf("  CPU time      : %.4f ms\n", ms);
    printf("  Throughput    : %.4f GB/s\n\n", (n/1e9)/(ms/1000.0));
    printf("  %-28s  Count\n", "Motif");
    printf("  %-28s  -----\n", "----------------------------");
    for (int p = 0; p < NUM_MOTIFS; p++)
        if (motifCounts[p] > 0)
            printf("  %-28s  %llu\n", MOTIF_NAMES[p], motifCounts[p]);

    cudaEventDestroy(s); cudaEventDestroy(e);
    free(ac); free(h_text);
    printf("\n[Done] AC CPU scan complete.\n");
    return 0;
}
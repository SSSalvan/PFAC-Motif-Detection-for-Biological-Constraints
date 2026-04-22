/**
 * AC.cpp  —  FIXED VERSION
 * =====================================================================
 * Sequential Aho-Corasick for DNA Biological Constraint Detection.
 *
 * Research: "Constraint-Aware PFAC Motif Matching on GPUs for
 *            High-Throughput DNA Storage Streams"
 * Reference: Gagniuc et al., 2025 — Algorithms 18(12), 742.
 *
 * What changed from previous version:
 *   - MOTIFS updated to real biological constraint sequences used in
 *     DNA data storage pipelines (restriction enzymes, homopolymers,
 *     repeats, G-quadruplex forming sequences).
 *   - Added motif category labels for reporting.
 *   - Timing uses clock_gettime for higher resolution.
 *   - File reading fixed for large inputs.
 *
 * Compile:
 *   g++ -O2 -o ac AC.cpp
 *   ./ac raw.txt
 *
 * Or with NVCC (no CUDA used here):
 *   nvcc -O2 -x c++ -o ac AC.cpp
 * =====================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/*  Constants                                                           */
/* ------------------------------------------------------------------ */
#define ALPHA_SIZE       4       // DNA: A=0, C=1, G=2, T=3
#define MAX_STATES    5000       // enough for all DNA motifs below
#define MAX_PATTERNS    64       // max biological constraint motifs
#define MAX_INPUT_LEN 1073741824 // 1 GB max (adjust as needed)

/* ------------------------------------------------------------------ */
/*  DNA base → integer index                                            */
/* ------------------------------------------------------------------ */
static int dna_map(char c) {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default:             return -1;  // non-ACGT → skip
    }
}

/* ------------------------------------------------------------------ */
/*  Aho-Corasick Automaton                                             */
/*  Formal definition (Gagniuc et al. 2025, Section 3.1):             */
/*    A = (Q, Σ, δ, q0, F, f, O)                                      */
/* ------------------------------------------------------------------ */
typedef struct {
    int                delta[MAX_STATES][ALPHA_SIZE]; // δ(q,a)
    int                f[MAX_STATES];                 // failure f(q)
    unsigned long long O[MAX_STATES];                 // output bitmask
    int                numStates;
} AhoCorasick;

/* ------------------------------------------------------------------ */
/*  Phase 1: Insert pattern into trie                                  */
/* ------------------------------------------------------------------ */
static void insertPattern(AhoCorasick *ac, const char *pattern, int patIndex) {
    int state = 0;
    for (int i = 0; pattern[i] != '\0'; i++) {
        int c = dna_map(pattern[i]);
        if (c == -1) continue;
        if (ac->delta[state][c] == -1) {
            ac->delta[state][c] = ac->numStates++;
        }
        state = ac->delta[state][c];
    }
    ac->O[state] |= (1ULL << patIndex);
}

/* ------------------------------------------------------------------ */
/*  Phase 2: Build failure links via BFS                               */
/*  Follows Algorithm 1 from Gagniuc et al. 2025, Section 3.2         */
/* ------------------------------------------------------------------ */
static void buildFailureLinks(AhoCorasick *ac) {
    int queue[MAX_STATES];
    int head = 0, tail = 0;

    // Depth-1 states: failure → root (q0)
    for (int c = 0; c < ALPHA_SIZE; c++) {
        if (ac->delta[0][c] != -1) {
            ac->f[ac->delta[0][c]] = 0;
            queue[tail++] = ac->delta[0][c];
        } else {
            ac->delta[0][c] = 0;  // undefined root transitions → root
        }
    }

    // BFS: compute failure links for all remaining states
    while (head < tail) {
        int v = queue[head++];
        for (int c = 0; c < ALPHA_SIZE; c++) {
            int u = ac->delta[v][c];
            if (u == -1) {
                // Pre-compute shortcut: δ(v,c) = δ(f(v),c)
                ac->delta[v][c] = ac->delta[ac->f[v]][c];
            } else {
                ac->f[u] = ac->delta[ac->f[v]][c];
                ac->O[u] |= ac->O[ac->f[u]];  // inherit output O(u) |= O(f(u))
                queue[tail++] = u;
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Biological Constraint Motifs for DNA Data Storage                  */
/*                                                                      */
/*  Categories (based on DNA storage constraint literature):           */
/*  1. HOMOPOLYMER RUNS   – synthesis error prone if ≥ 4 same bases   */
/*  2. RESTRICTION SITES  – enzyme recognition sites (cause cleavage)  */
/*  3. DINUCLEOTIDE REPS  – alternating repeats (structural instability)*/
/*  4. TANDEM REPEATS     – PCR slippage / polymerase errors           */
/*  5. GC-EXTREME RUNS    – melting temperature instability            */
/* ------------------------------------------------------------------ */

// Motif names (for reporting)
static const char *MOTIF_NAMES[] = {
    // ── Homopolymer Runs ────────────────────────────────────────────
    "HOMOPOLYMER_A4",    // AAAA   – poly-A ≥ 4 bases
    "HOMOPOLYMER_C4",    // CCCC   – poly-C ≥ 4 bases
    "HOMOPOLYMER_G4",    // GGGG   – poly-G ≥ 4 bases
    "HOMOPOLYMER_T4",    // TTTT   – poly-T ≥ 4 bases

    // ── Restriction Enzyme Recognition Sites ────────────────────────
    "ECORI_GAATTC",      // GAATTC – EcoRI  (most common in cloning)
    "BAMHI_GGATCC",      // GGATCC – BamHI
    "HINDIII_AAGCTT",    // AAGCTT – HindIII
    "XHOI_CTCGAG",       // CTCGAG – XhoI
    "SALI_GTCGAC",       // GTCGAC – SalI
    "NCOI_CCATGG",       // CCATGG – NcoI
    "NDEI_CATATG",       // CATATG – NdeI
    "SPHI_GCATGC",       // GCATGC – SphI

    // ── Dinucleotide Repeats (structural instability) ────────────────
    "REPEAT_ATATAT",     // ATATAT – AT repeat (B-DNA instability)
    "REPEAT_TATATA",     // TATATA – TA repeat
    "REPEAT_CGCGCG",     // CGCGCG – CpG repeat (methylation concern)
    "REPEAT_GCGCGC",     // GCGCGC – GC repeat

    // ── Short Tandem Repeats (PCR slippage) ─────────────────────────
    "STR_AAGAAG",        // AAGAAG
    "STR_CAGCAG",        // CAGCAG – CAG repeat (huntingtin-related)
    "STR_TGCTGC",        // TGCTGC

    // ── GC-Extreme Runs (melting temp instability) ───────────────────
    "GC_RUN_GCGCGCGC",   // GCGCGCGC – extreme GC run
    "AT_RUN_ATATATATAT", // ATATATATAT – extreme AT run
};

static const char *MOTIF_SEQS[] = {
    "AAAA",
    "CCCC",
    "GGGG",
    "TTTT",
    "GAATTC",
    "GGATCC",
    "AAGCTT",
    "CTCGAG",
    "GTCGAC",
    "CCATGG",
    "CATATG",
    "GCATGC",
    "ATATAT",
    "TATATA",
    "CGCGCG",
    "GCGCGC",
    "AAGAAG",
    "CAGCAG",
    "TGCTGC",
    "GCGCGCGC",
    "ATATATATAT",
};

static const int NUM_MOTIFS = 21;

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */
int main(int argc, char *argv[]) {

    const char *input_file = "raw.txt";
    if (argc > 1) input_file = argv[1];

    printf("========================================================\n");
    printf("  Sequential Aho-Corasick — DNA Constraint Detection    \n");
    printf("  Based on: Gagniuc et al., Algorithms 2025, 18, 742   \n");
    printf("========================================================\n\n");

    // ── Build Automaton ──────────────────────────────────────────────
    AhoCorasick *ac = (AhoCorasick *)malloc(sizeof(AhoCorasick));
    if (!ac) { fprintf(stderr, "Out of memory.\n"); return 1; }

    memset(ac->delta, -1, sizeof(ac->delta));
    memset(ac->f,      0, sizeof(ac->f));
    memset(ac->O,      0, sizeof(ac->O));
    ac->numStates = 1;

    printf("[AC-Build] Loading %d biological constraint motifs...\n", NUM_MOTIFS);
    for (int i = 0; i < NUM_MOTIFS; i++) {
        insertPattern(ac, MOTIF_SEQS[i], i);
        printf("  [%2d] %-24s  seq: %s\n", i, MOTIF_NAMES[i], MOTIF_SEQS[i]);
    }

    printf("\n[AC-Build] Building failure links (BFS)...\n");
    buildFailureLinks(ac);
    printf("[AC-Build] Done. Total automaton states: %d\n\n", ac->numStates);

    // ── Load DNA Sequence ────────────────────────────────────────────
    FILE *fp = fopen(input_file, "r");
    if (!fp) {
        printf("[ERROR] Could not open '%s'.\n", input_file);
        printf("        Run: python dna_loader.py   first!\n");
        free(ac);
        return 1;
    }

    char *text = (char *)malloc(MAX_INPUT_LEN);
    if (!text) { fprintf(stderr, "Out of memory for text.\n"); free(ac); return 1; }

    int n = (int)fread(text, 1, MAX_INPUT_LEN - 1, fp);
    fclose(fp);
    text[n] = '\0';

    printf("[AC-Search] Scanning %d bases sequentially...\n", n);

    // ── Sequential Scan ──────────────────────────────────────────────
    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    int state   = 0;
    int matches = 0;

    unsigned long long motif_counts[MAX_PATTERNS] = {0};

    for (int i = 0; i < n; i++) {
        int c = dna_map(text[i]);
        if (c == -1) continue;

        // Single O(1) transition (failure links pre-computed)
        state = ac->delta[state][c];

        if (ac->O[state] != 0ULL) {
            matches++;
            for (int p = 0; p < NUM_MOTIFS; p++) {
                if (ac->O[state] & (1ULL << p)) {
                    motif_counts[p]++;
                }
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double elapsed = (ts_end.tv_sec  - ts_start.tv_sec)
                   + (ts_end.tv_nsec - ts_start.tv_nsec) * 1e-9;

    // ── Results ──────────────────────────────────────────────────────
    printf("\n[Results]\n");
    printf("  Total matches found : %d\n", matches);
    printf("  Time elapsed        : %.6f seconds\n", elapsed);
    printf("  Throughput          : %.2f MB/s\n\n", (n / 1e6) / elapsed);

    printf("  Breakdown by motif category:\n");
    printf("  %-26s  %s\n", "Motif Name", "Count");
    printf("  %-26s  %s\n", "--------------------------", "-------");
    for (int p = 0; p < NUM_MOTIFS; p++) {
        if (motif_counts[p] > 0) {
            printf("  %-26s  %llu\n", MOTIF_NAMES[p], motif_counts[p]);
        }
    }

    free(ac);
    free(text);

    printf("\n[Done] Sequential AC scan complete.\n");
    return 0;
}
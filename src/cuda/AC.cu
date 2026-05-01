/**
 * AC.cu  —  Sequential Aho-Corasick (CPU Baseline)
 * Reference: Gagniuc et al., 2025 — Algorithms 18(12), 742.
 *
 * Perbaikan: Menambahkan Sequence Generator agar benchmark adil dengan PFAC.
 *            + Klasifikasi Kategori Motif (Constraint-Aware Output)
 * Compile: nvcc -O2 -o AC AC.cu
 * Run Real:    ./AC.exe raw.txt
 * Run Benchmark: ./AC.exe dummy.txt 65536
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

#define ALPHA_SIZE    4
#define MAX_STATES    20000
#define MAX_PATTERNS  64

/* ─── Definisi Kategori ─────────────────────────────────────────────────── */
typedef enum {
    CAT_HOMOPOLYMER  = 0,   /* synthesis hazard          → CRITICAL */
    CAT_RESTRICTION  = 1,   /* cleavage risk             → CRITICAL */
    CAT_DINUCLEOTIDE = 2,   /* structural instability    → HIGH     */
    CAT_STR          = 3,   /* PCR slippage              → MEDIUM   */
    CAT_GC_EXTREME   = 4,   /* melting temperature issue → LOW      */
    NUM_CATEGORIES   = 5
} MotifCategory;

static const char *CAT_NAMES[] = {
    "HOMOPOLYMER RUNS (synthesis hazard)",
    "RESTRICTION ENZYME SITES (cleavage risk)",
    "DINUCLEOTIDE REPEATS (structural instability)",
    "SHORT TANDEM REPEATS (PCR slippage)",
    "GC/AT EXTREME RUNS (melting temp)"
};

static const char *CAT_SEVERITY[] = {
    "CRITICAL", "CRITICAL", "HIGH", "MEDIUM", "LOW"
};

/* ─── Motif definitions ─────────────────────────────────────────────────── */
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

/* Mapping motif index → kategori (urutan sama dengan MOTIF_SEQS[]) */
static const MotifCategory MOTIF_CAT[] = {
    CAT_HOMOPOLYMER,   /* AAAA          */
    CAT_HOMOPOLYMER,   /* CCCC          */
    CAT_HOMOPOLYMER,   /* GGGG          */
    CAT_HOMOPOLYMER,   /* TTTT          */
    CAT_RESTRICTION,   /* GAATTC        */
    CAT_RESTRICTION,   /* GGATCC        */
    CAT_RESTRICTION,   /* AAGCTT        */
    CAT_RESTRICTION,   /* CTCGAG        */
    CAT_RESTRICTION,   /* GTCGAC        */
    CAT_RESTRICTION,   /* CCATGG        */
    CAT_RESTRICTION,   /* CATATG        */
    CAT_RESTRICTION,   /* GCATGC        */
    CAT_DINUCLEOTIDE,  /* ATATAT        */
    CAT_DINUCLEOTIDE,  /* TATATA        */
    CAT_DINUCLEOTIDE,  /* CGCGCG        */
    CAT_DINUCLEOTIDE,  /* GCGCGC        */
    CAT_STR,           /* AAGAAG        */
    CAT_STR,           /* CAGCAG        */
    CAT_STR,           /* TGCTGC        */
    CAT_GC_EXTREME,    /* GCGCGCGC      */
    CAT_GC_EXTREME,    /* ATATATATAT    */
};

/* ─── Generator DNA Random ──────────────────────────────────────────────── */
void generate_random_dna(char *seq, long long n) {
    const char bases[] = "ACGT";
    srand(42); /* Seed tetap agar data CPU dan GPU identik */
    for (long long i = 0; i < n; i++) {
        seq[i] = bases[rand() % 4];
    }
    seq[n] = '\0';
}

/* ─── Host helpers ──────────────────────────────────────────────────────── */
static int h_dna_map(char c) {
    switch(c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default: return -1;
    }
}

/* ─── Automaton ─────────────────────────────────────────────────────────── */
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

/* ─── Print hasil dengan klasifikasi kategori ───────────────────────────── */
static void printCategorizedResults(const unsigned long long *motifCounts) {
    printf("\n  %-28s  %s\n", "Motif", "Count");
    printf("  %-28s  -----\n", "----------------------------");

    unsigned long long grand_total = 0;

    for (int cat = 0; cat < NUM_CATEGORIES; cat++) {
        unsigned long long cat_total = 0;
        for (int p = 0; p < NUM_MOTIFS; p++)
            if ((int)MOTIF_CAT[p] == cat) cat_total += motifCounts[p];

        printf("\n  ── Category %d: %s ──\n", cat + 1, CAT_NAMES[cat]);

        for (int p = 0; p < NUM_MOTIFS; p++) {
            if ((int)MOTIF_CAT[p] == cat) {
                printf("  %-28s  %llu\n", MOTIF_NAMES[p], motifCounts[p]);
            }
        }

        printf("  %-28s  %llu  [%s]\n", "Subtotal:", cat_total, CAT_SEVERITY[cat]);
        grand_total += cat_total;
    }

    printf("\n  %-28s  %llu\n", "GRAND TOTAL:", grand_total);
}

/* ─── main ──────────────────────────────────────────────────────────────── */
int main(int argc, char *argv[]) {
    const char *input_file = "data/raw/raw.txt";
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

    char *h_text;
    long long n;

    FILE *fp = fopen(input_file, "r");
    if (!fp) {
        if (use_n > 0) {
            printf("[INFO] File %s not found. Generating %lld random bases (Research Guide Mode)...\n", input_file, use_n);
            n = use_n;
            h_text = (char *)malloc(n + 1);
            generate_random_dna(h_text, n);
        } else {
            printf("[ERROR] Cannot open %s and no size N provided.\n", input_file);
            return 1;
        }
    } else {
        fseek(fp, 0, SEEK_END);
        long long file_size = ftell(fp); fseek(fp, 0, SEEK_SET);
        n = (use_n > 0 && use_n < file_size) ? use_n : file_size;
        h_text = (char *)malloc(n + 1);
        fread(h_text, 1, n, fp);
        fclose(fp); h_text[n] = '\0';
    }

    printf("[AC] Scanning %lld bases...\n\n", n);
    unsigned long long motifCounts[MAX_PATTERNS] = {0};

    cudaEvent_t s_event, e_event;
    cudaEventCreate(&s_event); cudaEventCreate(&e_event);
    cudaEventRecord(s_event);

    int matches = searchAC(h_text, n, ac, motifCounts);

    cudaEventRecord(e_event); cudaEventSynchronize(e_event);
    float ms = 0; cudaEventElapsedTime(&ms, s_event, e_event);

    printf("[Results]\n");
    printf("  Total matches : %d\n", matches);
    printf("  CPU time      : %.4f ms\n", ms);
    printf("  Throughput    : %.4f GB/s\n", (n / 1e9) / (ms / 1000.0));

    printCategorizedResults(motifCounts);

    cudaEventDestroy(s_event); cudaEventDestroy(e_event);
    free(ac); free(h_text);
    printf("\n[Done] AC CPU scan complete.\n");
    return 0;
}
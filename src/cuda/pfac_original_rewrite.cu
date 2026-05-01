/**
 * pfac_original_rewrite.cu
 *
 * Clean rewrite of the sequential Aho-Corasick implementation.
 * Based on: Gagniuc et al., "The Aho-Corasick Paradigm in Modern
 *           Antivirus Engines", Algorithms 2025, 18, 742.
 *
 * Key differences from the original:
 *   - ALL large arrays are heap-allocated (no stack overflows on Windows)
 *   - No no-effect expressions or dead code
 *   - Proper cleanup of all allocations
 *   - No CUDA runtime dependency (pure CPU — no GPU needed for this file)
 *
 * Compilation:
 *   nvcc -O2 -allow-unsupported-compiler -Xcompiler "/std:c++14" -o ac_original pfac_original_rewrite.cu
 *   .\ac_original.exe
 *
 *   Or with plain cl.exe (no GPU needed):
 *   cl /O2 /Fe:ac_original.exe pfac_original_rewrite.cu
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Constants                                                           */
/* ------------------------------------------------------------------ */

#define ALPHABET_SIZE  256
#define MAX_STATES     10000
#define MAX_PATTERNS   64
#define MAX_PAT_LEN    128
#define MAX_INPUT_LEN  1048576   /* 1 MB */

/* ------------------------------------------------------------------ */
/*  Automaton — fully heap-allocated                                   */
/*                                                                      */
/*  sizeof(AhoCorasickAutomaton) ~ 10 MB — never put this on the stack */
/* ------------------------------------------------------------------ */

typedef struct {
    int                *delta;   /* [MAX_STATES * ALPHABET_SIZE] flattened  */
    int                *f;       /* [MAX_STATES] failure links              */
    unsigned long long *O;       /* [MAX_STATES] output bitmasks            */
    int                 numStates;
} AhoCorasickAutomaton;

/* Convenience macro for 2-D delta access */
#define DELTA(ac, s, c)  ((ac)->delta[(s) * ALPHABET_SIZE + (c)])

/* ------------------------------------------------------------------ */
/*  Allocate / free automaton                                           */
/* ------------------------------------------------------------------ */

static AhoCorasickAutomaton *acAlloc(void)
{
    AhoCorasickAutomaton *ac =
        (AhoCorasickAutomaton *)malloc(sizeof(AhoCorasickAutomaton));
    if (!ac) return NULL;

    ac->delta = (int *)malloc((size_t)MAX_STATES * ALPHABET_SIZE * sizeof(int));
    ac->f     = (int *)malloc((size_t)MAX_STATES * sizeof(int));
    ac->O     = (unsigned long long *)malloc((size_t)MAX_STATES * sizeof(unsigned long long));

    if (!ac->delta || !ac->f || !ac->O) {
        free(ac->delta); free(ac->f); free(ac->O); free(ac);
        return NULL;
    }
    return ac;
}

static void acFree(AhoCorasickAutomaton *ac)
{
    if (!ac) return;
    free(ac->delta);
    free(ac->f);
    free(ac->O);
    free(ac);
}

/* ------------------------------------------------------------------ */
/*  Utility — parse hex string to byte array                           */
/* ------------------------------------------------------------------ */

static int parseHexPattern(const char    *hexStr,
                            unsigned char *out,
                            int            maxLen)
{
    int len = 0;
    const char *p = hexStr;
    while (*p && len < maxLen) {
        while (*p == ' ') p++;
        if (!*p) break;
        unsigned int byte;
        if (sscanf(p, "%02x", &byte) == 1) {
            out[len++] = (unsigned char)byte;
            p += 2;
        } else break;
    }
    return len;
}

/* ------------------------------------------------------------------ */
/*  Phase 1 — Trie insertion                                           */
/* ------------------------------------------------------------------ */

static void insertPattern(AhoCorasickAutomaton *ac,
                           const unsigned char  *pattern,
                           int                   patLen,
                           int                   patIndex)
{
    int state = 0;
    for (int i = 0; i < patLen; i++) {
        unsigned char c = pattern[i];
        if (DELTA(ac, state, c) == -1)
            DELTA(ac, state, c) = ac->numStates++;
        state = DELTA(ac, state, c);
    }
    ac->O[state] |= (1ULL << patIndex);
}

/* ------------------------------------------------------------------ */
/*  Phase 2 — Failure links (BFS)                                      */
/* ------------------------------------------------------------------ */

static void buildFailureLinks(AhoCorasickAutomaton *ac)
{
    /* BFS queue on the heap — avoids 40 KB stack frame */
    int *queue = (int *)malloc((size_t)MAX_STATES * sizeof(int));
    if (!queue) { fprintf(stderr, "OOM (queue)\n"); return; }
    int head = 0, tail = 0;

    /* Depth-1 states: failure → root */
    for (int c = 0; c < ALPHABET_SIZE; c++) {
        if (DELTA(ac, 0, c) != -1) {
            int s = DELTA(ac, 0, c);
            ac->f[s] = 0;
            queue[tail++] = s;
        } else {
            DELTA(ac, 0, c) = 0;   /* undefined root transitions → root */
        }
    }

    /* BFS over remaining states */
    while (head < tail) {
        int v = queue[head++];
        for (int c = 0; c < ALPHABET_SIZE; c++) {
            int u = DELTA(ac, v, c);
            if (u == -1) {
                /* pre-compute failure shortcut */
                DELTA(ac, v, c) = DELTA(ac, ac->f[v], c);
                continue;
            }
            ac->f[u] = DELTA(ac, ac->f[v], c);
            ac->O[u] |= ac->O[ac->f[u]];   /* inherit output */
            queue[tail++] = u;
        }
    }

    free(queue);
}

/* ------------------------------------------------------------------ */
/*  Build entry point                                                   */
/* ------------------------------------------------------------------ */

static void acBuild(AhoCorasickAutomaton     *ac,
                    const unsigned char      *patterns[],
                    const int                 patLens[],
                    int                       numPatterns)
{
    /* initialise all transitions to -1 */
    memset(ac->delta, -1, (size_t)MAX_STATES * ALPHABET_SIZE * sizeof(int));
    memset(ac->f,      0, (size_t)MAX_STATES * sizeof(int));
    memset(ac->O,      0, (size_t)MAX_STATES * sizeof(unsigned long long));
    ac->numStates = 1;   /* state 0 = root q0 */

    for (int i = 0; i < numPatterns; i++)
        insertPattern(ac, patterns[i], patLens[i], i);

    buildFailureLinks(ac);
}

/* ------------------------------------------------------------------ */
/*  Sequential scan                                                     */
/* ------------------------------------------------------------------ */

static int acSearch(const AhoCorasickAutomaton *ac,
                    const unsigned char        *text,
                    int                         textLen,
                    const char                 *patternNames[],
                    int                         numPatterns)
{
    int state      = 0;
    int matchCount = 0;

    printf("\n[AC-Search] Scanning %d bytes sequentially...\n", textLen);

    for (int i = 0; i < textLen; i++) {
        unsigned char c = text[i];
        state = DELTA(ac, state, c);

        if (ac->O[state] != 0ULL) {
            for (int p = 0; p < numPatterns; p++) {
                if (ac->O[state] & (1ULL << p)) {
                    printf("  [MATCH] Pattern %-20s found ending at byte offset %d\n",
                           patternNames[p], i);
                    matchCount++;
                }
            }
        }
    }

    printf("[AC-Search] Done. Total matches: %d\n", matchCount);
    return matchCount;
}

/* ------------------------------------------------------------------ */
/*  Main                                                                */
/* ------------------------------------------------------------------ */

int main(void)
{
    printf("========================================================\n");
    printf("  Aho-Corasick Original (Sequential) — Clean Rewrite    \n");
    printf("  Based on: Gagniuc et al., Algorithms 2025, 18, 742    \n");
    printf("========================================================\n\n");

    /* ---- Signatures ------------------------------------------- */
    const char *sigNames[] = {
        "P1=25_26_33",
        "P2=3B_35_34",
        "P3=33_35_3C",
        "P4=7B_54_49_39"
    };
    const char *sigHex[] = {
        "25 26 33",
        "3B 35 34",
        "33 35 3C",
        "7B 54 49 39"
    };
    int numSigs = 4;

    /* patternBytes on heap — avoids stack frame bloat */
    unsigned char (*patternBytes)[MAX_PAT_LEN] =
        (unsigned char (*)[MAX_PAT_LEN])malloc(MAX_PATTERNS * MAX_PAT_LEN);
    const unsigned char *patterns[MAX_PATTERNS];
    int patLens[MAX_PATTERNS];

    if (!patternBytes) { fprintf(stderr, "OOM\n"); return 1; }

    for (int i = 0; i < numSigs; i++) {
        patLens[i]  = parseHexPattern(sigHex[i], patternBytes[i], MAX_PAT_LEN);
        patterns[i] = patternBytes[i];
        printf("Loaded signature [%s] (%d bytes)\n", sigNames[i], patLens[i]);
    }

    /* ---- Build automaton -------------------------------------- */
    AhoCorasickAutomaton *ac = acAlloc();
    if (!ac) { fprintf(stderr, "OOM\n"); free(patternBytes); return 1; }

    printf("\n[AC-Build] Constructing automaton from %d signatures...\n", numSigs);
    acBuild(ac, patterns, patLens, numSigs);
    printf("[AC-Build] Done. Total states: %d\n", ac->numStates);

    /* ---- Input stream ----------------------------------------- */
    const char *inputHex =
        "33 3B 3B 35 34 35 38 35 3B 37 45 32 44 3B 25 26 33";

    unsigned char *inputStream = (unsigned char *)malloc(MAX_INPUT_LEN);
    if (!inputStream) {
        fprintf(stderr, "OOM\n");
        acFree(ac); free(patternBytes); return 1;
    }
    int inputLen = parseHexPattern(inputHex, inputStream, MAX_INPUT_LEN);
    printf("\nInput stream (%d bytes): %s\n", inputLen, inputHex);

    /* ---- Scan ------------------------------------------------- */
    acSearch(ac, inputStream, inputLen, sigNames, numSigs);

    /* ---- Cleanup ---------------------------------------------- */
    free(inputStream);
    free(patternBytes);
    acFree(ac);

    printf("\n[Done] Sequential AC scan complete.\n");
    return 0;
}

/**
 * aho_corasick_original.cu
 *
 * Original / Reference Aho-Corasick Implementation
 * Based on: Gagniuc, P.A.; Păvăloiu, I.-B.; Dascălu, M.-I.
 *           "The Aho-Corasick Paradigm in Modern Antivirus Engines"
 *           Algorithms 2025, 18, 742.
 *
 * This file implements the classic, sequential Aho-Corasick automaton
 * as described in the paper (Algorithm 1). The automaton is built on
 * the CPU and scanning is performed sequentially — one byte at a time —
 * exactly matching the formal description in Section 3.1 of the paper.
 *
 * Complexity:
 *   Build : O(sum(|pi|))
 *   Search: O(m + z)   where m = input length, z = number of matches
 *   Space : O(sum(|pi|) * |Sigma|)  worst case (dense transition table)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Constants                                                           */
/* ------------------------------------------------------------------ */

#define ALPHABET_SIZE 256          /* byte-level alphabet (hex stream)  */
#define MAX_STATES    10000        /* max trie nodes                     */
#define MAX_PATTERNS  64           /* max number of signatures           */
#define MAX_PAT_LEN   128          /* max bytes per signature            */
#define MAX_INPUT_LEN 1048576      /* max scanned stream length (1 MB)   */

/* ------------------------------------------------------------------ */
/*  Data Structures                                                     */
/* ------------------------------------------------------------------ */

/*
 * AhoCorasickAutomaton
 * Matches the formal definition from the paper:
 *   A = (Q, Sigma, delta, q0, F, f, O)
 *
 *   Q      : states 0..numStates-1  (q0 = root = 0)
 *   delta  : goto/transition table  (explicit, dense — classic AC)
 *   f      : failure function
 *   O      : output function (bitmask over pattern indices)
 */
typedef struct {
    int  delta[MAX_STATES][ALPHABET_SIZE];  /* transition function δ(q, a) */
    int  f[MAX_STATES];                     /* failure function f(q)        */
    unsigned long long O[MAX_STATES];       /* output bitmask per state     */
    int  numStates;                         /* total states built           */
} AhoCorasickAutomaton;

/* ------------------------------------------------------------------ */
/*  Step 1 — Trie insertion  (AC-Build, phase 1)                       */
/* ------------------------------------------------------------------ */

/**
 * insertPattern
 * Inserts one pattern into the trie, creating new states as needed.
 * Terminal state is tagged with its pattern index in the output bitmask.
 */
static void insertPattern(AhoCorasickAutomaton *ac,
                          const unsigned char  *pattern,
                          int                   patLen,
                          int                   patIndex)
{
    int state = 0;  /* start from root q0 */

    for (int i = 0; i < patLen; i++) {
        unsigned char c = pattern[i];
        if (ac->delta[state][c] == -1) {
            /* no transition on c → create new state */
            /* (new state zero-initialised at alloc — no action needed here) */
            ac->delta[state][c] = ac->numStates++;
        }
        state = ac->delta[state][c];
    }

    /* mark this state as accepting for pattern patIndex */
    ac->O[state] |= (1ULL << patIndex);
}

/* ------------------------------------------------------------------ */
/*  Step 2 — Failure links  (AC-Build, phase 2 — BFS)                 */
/* ------------------------------------------------------------------ */

/**
 * buildFailureLinks
 * Computes failure (fall-back) transitions via BFS, exactly as
 * described in Algorithm 1 of the paper.
 * After BFS, output sets are propagated: O(u) |= O(f(u)).
 */
static void buildFailureLinks(AhoCorasickAutomaton *ac)
{
    /* BFS queue — at most MAX_STATES entries */
    int queue[MAX_STATES];
    int head = 0, tail = 0;

    /* Depth-1 states: failure → root (q0 = 0) */
    for (int c = 0; c < ALPHABET_SIZE; c++) {
        if (ac->delta[0][c] != -1) {
            int s = ac->delta[0][c];
            ac->f[s] = 0;
            queue[tail++] = s;
        } else {
            /* optional fallback shortcut: undefined root transitions → root */
            ac->delta[0][c] = 0;
        }
    }

    /* BFS over remaining states */
    while (head < tail) {
        int v = queue[head++];

        for (int c = 0; c < ALPHABET_SIZE; c++) {
            int u = ac->delta[v][c];
            if (u == -1) {
                /*
                 * No explicit transition: shortcut δ(v,c) = δ(f(v),c)
                 * This pre-computes the "follow failure links" step,
                 * giving O(1) per symbol during search.
                 */
                ac->delta[v][c] = ac->delta[ac->f[v]][c];
                continue;
            }

            /* u is a real child: compute its failure link */
            int x = ac->f[v];
            /* follow failure links until a transition on c exists */
            /* (already resolved because BFS processes nodes in order) */
            ac->f[u] = ac->delta[x][c];

            /* inherit output: O(u) |= O(f(u)) */
            ac->O[u] |= ac->O[ac->f[u]];

            queue[tail++] = u;
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Step 3 — Automaton construction entry point                        */
/* ------------------------------------------------------------------ */

/**
 * acBuild
 * Initialises the automaton and runs both phases of AC-Build.
 */
static void acBuild(AhoCorasickAutomaton     *ac,
                    const unsigned char      *patterns[],
                    const int                 patLens[],
                    int                       numPatterns)
{
    /* initialise all transitions to -1 (undefined) */
    memset(ac->delta, -1, sizeof(ac->delta));
    memset(ac->f,      0, sizeof(ac->f));
    memset(ac->O,      0, sizeof(ac->O));
    ac->numStates = 1;  /* state 0 = root q0 */

    /* Phase 1: insert every pattern into the trie */
    for (int i = 0; i < numPatterns; i++) {
        insertPattern(ac, patterns[i], patLens[i], i);
    }

    /* Phase 2: build failure links via BFS */
    buildFailureLinks(ac);
}

/* ------------------------------------------------------------------ */
/*  Step 4 — Sequential search  (AC-Search)                           */
/* ------------------------------------------------------------------ */

/**
 * acSearch
 * Scans the input stream sequentially, one byte at a time.
 * Matches the pseudocode in Algorithm 1 (paper, Section 3.2).
 *
 * For every match, prints the pattern index and byte offset.
 * Returns total number of matches found.
 */
static int acSearch(const AhoCorasickAutomaton *ac,
                    const unsigned char        *text,
                    int                         textLen,
                    const char                 *patternNames[],
                    int                         numPatterns)
{
    int state      = 0;  /* current automaton state, starts at q0 */
    int matchCount = 0;

    printf("\n[AC-Search] Scanning %d bytes sequentially...\n", textLen);

    for (int i = 0; i < textLen; i++) {
        unsigned char c = text[i];

        /*
         * Transition: because buildFailureLinks pre-computes shortcuts,
         * δ(state, c) is always defined — O(1) per symbol.
         */
        state = ac->delta[state][c];

        /* check output function O(state) */
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
/*  Utility — build a byte pattern from a hex string                   */
/* ------------------------------------------------------------------ */

/**
 * parseHexPattern
 * Converts a space-separated hex string like "25 26 33" into bytes.
 * Returns the number of bytes parsed.
 */
static int parseHexPattern(const char *hexStr,
                            unsigned char *out,
                            int maxLen)
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
        } else {
            break;
        }
    }
    return len;
}

/* ------------------------------------------------------------------ */
/*  Main — demonstration with signatures from Figure 2 of the paper    */
/* ------------------------------------------------------------------ */

int main(void)
{
    printf("========================================================\n");
    printf("  Aho-Corasick Original (Sequential) — Reference Impl.  \n");
    printf("  Based on: Gagniuc et al., Algorithms 2025, 18, 742    \n");
    printf("========================================================\n\n");

    /*
     * Signatures from Figure 2 of the paper (hex byte sequences).
     * In a real antivirus engine these come from a .db file.
     * Format: name = hex_pattern  (Appendix A.1 of the paper)
     */
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

    /* parse hex strings into byte arrays */
    unsigned char patternBytes[MAX_PATTERNS][MAX_PAT_LEN];
    const unsigned char *patterns[MAX_PATTERNS];
    int patLens[MAX_PATTERNS];

    for (int i = 0; i < numSigs; i++) {
        patLens[i] = parseHexPattern(sigHex[i], patternBytes[i], MAX_PAT_LEN);
        patterns[i] = patternBytes[i];
        printf("Loaded signature [%s] (%d bytes)\n", sigNames[i], patLens[i]);
    }

    /* -------------------------------------------------------------- */
    /*  Build the automaton (CPU, sequential)                          */
    /* -------------------------------------------------------------- */
    AhoCorasickAutomaton *ac =
        (AhoCorasickAutomaton *)malloc(sizeof(AhoCorasickAutomaton));
    if (!ac) { fprintf(stderr, "Out of memory.\n"); return 1; }

    printf("\n[AC-Build] Constructing automaton from %d signatures...\n", numSigs);
    acBuild(ac, patterns, patLens, numSigs);
    printf("[AC-Build] Done. Total states: %d\n", ac->numStates);

    /* -------------------------------------------------------------- */
    /*  Simulated binary input stream (hex bytes from Figure 2)        */
    /*  Input: 33 3B 3B 35 34 35 38 35 3B 37 45 32 44 3B 25 26 33     */
    /* -------------------------------------------------------------- */
    const char *inputHex =
        "33 3B 3B 35 34 35 38 35 3B 37 45 32 44 3B 25 26 33";
    unsigned char inputStream[MAX_INPUT_LEN];
    int inputLen = parseHexPattern(inputHex, inputStream, MAX_INPUT_LEN);

    printf("\nInput stream (%d bytes): %s\n", inputLen, inputHex);

    /* -------------------------------------------------------------- */
    /*  Sequential scan — one byte at a time, single CPU thread        */
    /* -------------------------------------------------------------- */
    acSearch(ac, inputStream, inputLen, sigNames, numSigs);

    free(ac);
    printf("\n[Done] Sequential AC scan complete.\n");
    return 0;
}

/*
 * Compilation (host-only, no GPU needed for this file):
 *   nvcc -o ac_original aho_corasick_original.cu
 *   ./ac_original
 *
 * Or with plain GCC (since no CUDA kernels are used here):
 *   gcc -o ac_original aho_corasick_original.cu -lm
 *   ./ac_original
 */

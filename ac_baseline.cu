/**
 * ============================================================================
 * ac_baseline.cu — CPU Aho-Corasick Baseline Implementation
 * ============================================================================
 * Paper : "String Matching Performance and Scalability Analysis of
 *          Aho-Corasick and PFAC Algorithms on CUDA GPUs for DNA Sequences"
 *
 * AC Baseline Reference:
 *   Gagniuc, P.A. et al., "The Aho-Corasick Paradigm in Modern Antivirus
 *   Engines: A Cornerstone of Signature-Based Malware Detection,"
 *   Algorithms 2025, 18, 742. [Ref 5]
 *
 * Algorithm:
 *   Classic AC — builds a finite automaton (trie + failure links) from a
 *   pattern set and traverses the input text in a single linear pass.
 *   Time complexity : O(n + m + z)  where n=text length, m=total pattern
 *                     length, z=number of matches  [Aho & Corasick, 1975]
 *   Space complexity: O(|Σ| × Q)   where |Σ|=4 (DNA), Q=number of states
 *
 * Datasets (place in data/genomic/ relative to executable):
 *   Small  : knownCanonical.exonNuc.fa   (human exons, hg38)
 *   Medium : CHM13v2.0_genomic.fna       (T2T human genome, capped)
 *   Large  : dmel-all-aligned-r6.66.fasta (D. melanogaster)
 *   XSmall : cere/strains/S288c/assembly/genome.fa (S. cerevisiae)
 *
 * Metrics reported:
 *   Comparison : Execution Time, Throughput, Memory Usage, Matches
 *   Scalability: across 5 input sizes and motif counts
 *
 * Build:
 *   nvcc -O3 -arch=sm_89 --allow-unsupported-compiler ac_baseline.cu -o ac_baseline
 *   ac_baseline.exe
 * ============================================================================
 */

#include <cuda_runtime.h>   /* for cudaGetDeviceProperties only */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>

/* ── Cross-platform timer ── */
#ifdef _WIN32
#  include <windows.h>
   typedef LARGE_INTEGER ts_t;
   static LARGE_INTEGER _qpf; static int _qi=0;
   static inline void   tnow(ts_t *t){ if(!_qi){QueryPerformanceFrequency(&_qpf);_qi=1;} QueryPerformanceCounter(t); }
   static inline double tsec(ts_t *a,ts_t *b){ return (double)(b->QuadPart-a->QuadPart)/(double)_qpf.QuadPart; }
#else
#  include <time.h>
   typedef struct timespec ts_t;
   static inline void   tnow(ts_t *t){ clock_gettime(CLOCK_MONOTONIC,t); }
   static inline double tsec(ts_t *a,ts_t *b){ return (b->tv_sec-a->tv_sec)+(b->tv_nsec-a->tv_nsec)*1e-9; }
#endif

/* ── Constants ── */
#define DNA_ALPHA    4        /* {A=0, C=1, G=2, T=3}                        */
#define MAX_PATS   512
#define MAX_PAT_LEN 18        /* k-mer length                                 */
#define MAX_STATES (MAX_PATS * MAX_PAT_LEN * 3 + 2000)
#define MAX_MATCHES (1 << 23)

/* Dataset paths */
#define PATH_EXON   "data/genomic/knownCanonical.exonNuc.fa/knownCanonical.exonNuc.fa"
#define PATH_CHM13  "data/genomic/CHM13v2.0_genomic.fna/CHM13v2.0_genomic.fna"
#define PATH_DMEL   "data/genomic/dmel-all-aligned-r6.66.fasta/dmel-all-aligned-r6.66.fasta"
#define PATH_YEAST  "data/genomic/cere/strains/S288c/assembly/genome.fa"
#define CAP_MB       512UL

/* ── DNA encoder ── */
static inline int dna_enc(uint8_t c){
    switch(c){
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default:             return -1;
    }
}

/* ── Structs ── */
typedef struct {
    int next[DNA_ALPHA];
    int fail;
    int out;    /* pattern id if terminal, else -1 */
    int depth;
} ACNode;

typedef struct { int pos; int pat; } Match;

typedef struct {
    double exec_ms;
    double throughput_mbps;
    size_t mem_bytes;
    int    matches;
} ACResult;

typedef struct {
    int    np;
    size_t sz;
    double exec_ms;
    double throughput_mbps;
} ScalePt;

/* =========================================================================
 * GLOBALS
 * ========================================================================= */
static ACNode  g_nodes[MAX_STATES];
static int     g_ns = 0;
static uint8_t g_pats[MAX_PATS][MAX_PAT_LEN];
static int     g_pl  [MAX_PATS];
static int     g_np  = 0;
static Match   g_matches[MAX_MATCHES];
static Match   g_tmp    [MAX_MATCHES];

/* =========================================================================
 * FASTA LOADER
 * Encodes DNA to 0-3, stores 0xFF for non-DNA (N, gaps, etc.)
 * ========================================================================= */
static uint8_t *load_fasta(const char *path, size_t *len, size_t cap)
{
    FILE *f = fopen(path, "r");
    if(!f){ fprintf(stderr,"[WARN] Cannot open: %s\n", path); *len=0; return NULL; }
    fseek(f,0,SEEK_END); long fsz=ftell(f); rewind(f);
    size_t alloc = (cap>0 && (size_t)fsz>cap) ? cap+4096 : (size_t)fsz+4096;
    uint8_t *buf = (uint8_t*)malloc(alloc+1);
    if(!buf){ fprintf(stderr,"[ERROR] malloc failed for %s\n",path); fclose(f); *len=0; return NULL; }
    size_t pos=0; char line[8192];
    while(fgets(line,sizeof(line),f)){
        if(line[0]=='>') continue;
        for(int i=0; line[i]&&line[i]!='\n'&&line[i]!='\r'; i++){
            if(cap>0 && pos>=cap) goto done;
            int e = dna_enc((uint8_t)line[i]);
            buf[pos++] = (e>=0) ? (uint8_t)e : 0xFF;
        }
    }
done:
    fclose(f); buf[pos]='\0'; *len=pos;
    return buf;
}

/* =========================================================================
 * AC AUTOMATON BUILD  (Gagniuc et al. 2025, Algorithm 1)
 * ========================================================================= */
static int ac_new(void){
    if(g_ns >= MAX_STATES){
        fprintf(stderr,"[ERROR] MAX_STATES exceeded.\n"); exit(1);
    }
    int id=g_ns++;
    memset(g_nodes[id].next,-1,sizeof(g_nodes[id].next));
    g_nodes[id].fail=0; g_nodes[id].out=-1; g_nodes[id].depth=0;
    return id;
}

static void ac_insert(const uint8_t *p, int len, int pid){
    int cur=0;
    for(int i=0;i<len;i++){
        int c = (int)p[i];
        if(c == 0xFF) continue;                    /* non-DNA */
        if(c >= 4){                                /* still ASCII? */
            c = dna_enc((uint8_t)c);
            if(c < 0) continue;
        }
        if(g_nodes[cur].next[c]==-1){
            int n=ac_new();
            g_nodes[n].depth=g_nodes[cur].depth+1;
            g_nodes[cur].next[c]=n;
        }
        cur=g_nodes[cur].next[c];
    }
    g_nodes[cur].out=pid;
}

/* BFS to compute failure links and propagate output functions */
static void ac_build_failure(void){
    static int q[MAX_STATES]; int h=0,t=0;
    /* depth-1 nodes: failure → root */
    for(int c=0;c<DNA_ALPHA;c++){
        int s=g_nodes[0].next[c];
        if(s==-1) g_nodes[0].next[c]=0;   /* root shortcut */
        else{ g_nodes[s].fail=0; q[t++]=s; }
    }
    while(h<t){
        int v=q[h++];
        /* inherit output through failure chain */
        if(g_nodes[v].out==-1) g_nodes[v].out=g_nodes[g_nodes[v].fail].out;
        for(int c=0;c<DNA_ALPHA;c++){
            int u=g_nodes[v].next[c];
            if(u==-1){
                /* fill shortcut via failure */
                g_nodes[v].next[c]=g_nodes[g_nodes[v].fail].next[c];
            } else {
                int x=g_nodes[v].fail;
                while(x && g_nodes[x].next[c]==-1) x=g_nodes[x].fail;
                int fn=g_nodes[x].next[c]; if(fn==u) fn=0;
                g_nodes[u].fail=fn; q[t++]=u;
            }
        }
    }
}

/* =========================================================================
 * AC SCAN  — single-threaded linear traversal (Gagniuc et al. 2025)
 * Processes each DNA symbol exactly once. O(n + z) per scan.
 * ========================================================================= */
static int ac_scan(const uint8_t *data, int len, Match *out, int mx)
{
    int st=0, n=0;
    for(int i=0; i<len && n<mx; i++){
        int c = (int)data[i];
        if(c == 0xFF) continue;         /* skip non-DNA */
        st = g_nodes[st].next[c];
        if(st < 0) st = 0;             /* safety: should not happen after ac_build */
        if(g_nodes[st].out != -1){
            out[n].pos = i;
            out[n].pat = g_nodes[st].out;
            n++;
        }
    }
    return n;
}

/* =========================================================================
 * REPORT
 * ========================================================================= */
static void sep(char c, int w){ for(int i=0;i<w;i++) putchar(c); putchar('\n'); }

static void report(const ACResult *r, const ScalePt *sp, int nsp,
                   int np, size_t total)
{
    const int W=72;
    printf("\n"); sep('=',W);
    printf("  AC BASELINE — BENCHMARK RESULTS\n");
    printf("  Ref: Gagniuc et al., Algorithms 2025, 18, 742 [Ref 5]\n");
    sep('=',W);
    printf("  Algorithm    : Aho-Corasick (CPU, single-threaded)\n");
    printf("  Automaton    : %d states, DNA 4-symbol alphabet\n", g_ns);
    printf("  Table size   : %d states × 4 × 4B = %d KB\n",
           g_ns, g_ns*DNA_ALPHA*4/1024);
    printf("  Patterns     : %d k-mers (k=%d nt)\n", np, MAX_PAT_LEN);
    printf("  Input        : %.3f MB (%zu bytes)\n\n",
           (double)total/1e6, total);

    sep('-',W);
    printf("  COMPARISON METRICS (CPU AC)\n"); sep('-',W);
    printf("  %-30s  %12.3f ms\n",  "1. Execution Time",    r->exec_ms);
    printf("  %-30s  %12.2f MB/s\n","2. Throughput",        r->throughput_mbps);
    printf("  %-30s  %12s\n",       "3. SpeedUp",           "1.00x (baseline)");
    printf("  %-30s  %12.2f MB\n",  "4. Memory Usage",      (double)r->mem_bytes/1e6);
    printf("  %-30s  %12d\n",       "5. Matches Found",     r->matches);
    printf("  %-30s  %12s\n",       "   Accuracy",          "100%% (exact AC scan)");

    printf("\n  6. Scalability:\n");
    printf("     %-8s  %-12s  %12s  %12s\n",
           "Motifs","Input(MB)","Exec(ms)","Throughput");
    for(int i=0;i<nsp;i++)
        printf("     %-8d  %-12.3f  %12.3f  %9.2f MB/s\n",
               sp[i].np, (double)sp[i].sz/1e6,
               sp[i].exec_ms, sp[i].throughput_mbps);

    printf("\n"); sep('-',W);
    printf("  NOTE: AC is the CPU baseline. Compare these results\n");
    printf("        against pfac_gpu.exe output for speedup analysis.\n");
    sep('=',W); printf("\n");
}

/* =========================================================================
 * MAIN
 * ========================================================================= */
int main(void)
{
    printf("============================================================\n");
    printf("  AC BASELINE — Aho-Corasick CPU Implementation\n");
    printf("  Ref: Gagniuc et al., Algorithms 2025, 18, 742\n");
    printf("============================================================\n\n");

    /* Load datasets */
    printf("Loading FASTA datasets...\n"); fflush(stdout);
    size_t l1=0,l2=0,l3=0,l4=0;
    printf("  [1/4] Human exons (hg38)...       "); fflush(stdout);
    uint8_t *d1=load_fasta(PATH_EXON,  &l1, CAP_MB*1024*1024);
    printf("%.3f MB\n",(double)l1/1e6);
    printf("  [2/4] Human T2T CHM13...          "); fflush(stdout);
    uint8_t *d2=load_fasta(PATH_CHM13, &l2, CAP_MB*1024*1024);
    printf("%.3f MB\n",(double)l2/1e6);
    printf("  [3/4] D. melanogaster r6.66...    "); fflush(stdout);
    uint8_t *d3=load_fasta(PATH_DMEL,  &l3, CAP_MB*1024*1024);
    printf("%.3f MB\n",(double)l3/1e6);
    printf("  [4/4] S. cerevisiae S288c...      "); fflush(stdout);
    uint8_t *d4=load_fasta(PATH_YEAST, &l4, 0);
    printf("%.3f MB\n\n",(double)l4/1e6); fflush(stdout);

    size_t total = l1+l2+l3+l4;
    if(total==0){
        fprintf(stderr,"[ERROR] No data loaded. Check paths in source.\n"); return 1;
    }

    uint8_t *hdata = (uint8_t*)malloc(total+1);
    size_t wp=0;
    if(d1){memcpy(hdata+wp,d1,l1);wp+=l1;free(d1);}
    if(d2){memcpy(hdata+wp,d2,l2);wp+=l2;free(d2);}
    if(d3){memcpy(hdata+wp,d3,l3);wp+=l3;free(d3);}
    if(d4){memcpy(hdata+wp,d4,l4);wp+=l4;free(d4);}
    hdata[wp]='\0';

    /* Extract k-mer motifs */
    int stride=(int)(total/MAX_PATS)+1;
    g_np = 0;
    for(size_t i=0; i<total-(size_t)MAX_PAT_LEN && g_np<MAX_PATS; i+=stride){
        int ok=1;
        for(int j=0;j<MAX_PAT_LEN;j++)
            if(hdata[i+j]==0xFF){ok=0;break;}
        if(ok){ 
            memcpy(g_pats[g_np],hdata+i,MAX_PAT_LEN); 
            g_pl[g_np]=MAX_PAT_LEN; 
            g_np++; 
        }
    }
    printf("  Total input : %.3f MB\n", (double)total/1e6);
    printf("  Motifs      : %d (k=%d nt, stride=%d)\n\n", g_np, MAX_PAT_LEN, stride);

    /* Build automaton */
    printf("Building AC automaton (failure links via BFS)...\n"); fflush(stdout);
    g_ns=0; ac_new();
    for(int i=0;i<g_np;i++) ac_insert(g_pats[i],g_pl[i],i);
    ac_build_failure();
    printf("  States: %d  |  Table: %d KB\n\n", g_ns, g_ns*DNA_ALPHA*4/1024); fflush(stdout);

    /* Main scan */
    printf("Running AC scan...\n"); fflush(stdout);
    ACResult res; memset(&res,0,sizeof(res));
    res.mem_bytes = (size_t)g_ns*sizeof(ACNode) + total;
    ts_t t0,t1;
    tnow(&t0);
    res.matches = ac_scan(hdata,(int)total,g_matches,MAX_MATCHES);
    tnow(&t1);
    double s = tsec(&t0,&t1);
    res.exec_ms        = s*1000.0;
    res.throughput_mbps = (double)total/1e6/s;
    printf("  Done: %d matches in %.3f ms  (%.2f MB/s)\n\n",
           res.matches, res.exec_ms, res.throughput_mbps); fflush(stdout);

    /* Scalability sweep */
    printf("Scalability sweep (5 points)...\n"); fflush(stdout);
    const double pf[]={0.10,0.25,0.50,0.75,1.00};
    const double sf[]={0.05,0.15,0.30,0.60,1.00};
    ScalePt sp[5];
    for(int si=0;si<5;si++){
        int snp=(int)(g_np*pf[si]); if(snp<1)snp=1;
        size_t ssz=(size_t)(total*sf[si]); if(ssz<1024)ssz=1024;
        /* rebuild subset automaton */
        int save=g_np; g_np=snp;
        g_ns=0; ac_new();
        for(int i=0;i<snp;i++) ac_insert(g_pats[i],g_pl[i],i);
        ac_build_failure();
        tnow(&t0);
        ac_scan(hdata,(int)ssz,g_tmp,MAX_MATCHES);
        tnow(&t1);
        double sc=tsec(&t0,&t1);
        sp[si].np=snp; sp[si].sz=ssz;
        sp[si].exec_ms=sc*1000.0;
        sp[si].throughput_mbps=(double)ssz/1e6/sc;
        printf("  [%d/5] %3d motifs × %.2f MB : %.3f ms  (%.1f MB/s)\n",
               si+1,snp,(double)ssz/1e6,sp[si].exec_ms,sp[si].throughput_mbps);
        fflush(stdout);
        g_np=save;
    }
    /* restore */
    g_ns=0; ac_new();
    for(int i=0;i<g_np;i++) ac_insert(g_pats[i],g_pl[i],i);
    ac_build_failure();

    report(&res,sp,5,g_np,total);
    free(hdata);
    return 0;
}

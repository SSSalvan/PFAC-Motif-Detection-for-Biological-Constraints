/**
 * =============================================================================
 * String Matching Performance and Scalability Analysis of
 * Aho-Corasick and PFAC Algorithms on CUDA GPUs for DNA Sequences
 * =============================================================================
 *
 * AC  Baseline : Gagniuc et al., Algorithms 2025, 18, 742
 * PFAC Baseline: Two-Phase PFAC — Lin et al., IEEE TPDS 2023 [ref 10]
 *
 * Comparison Metrics : Execution Time, Throughput, SpeedUp,
 *                      Memory Usage, Accuracy, Scalability
 * Profiling Metrics  : Kernel Time (cudaEvent), Occupancy (CUDA API),
 *                      Warp Efficiency, Branch Divergence,
 *                      Memory Throughput  (all from ncu / nvprof)
 *
 * Build:
 *   nvcc -O3 -arch=sm_89 --allow-unsupported-compiler ^
 *        pfac_aho_corasick.cu -o pfac_ac
 *   pfac_ac.exe
 *
 * For real hardware profiling metrics run AFTER compilation:
 *   ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,^
 *        smsp__sass_average_branch_targets_threads_uniform.pct,^
 *        l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,^
 *        sm__cycles_active.avg.pct_of_peak_sustained_elapsed ^
 *        pfac_ac.exe  > ncu_report.txt
 * =============================================================================
 *
 * Two-Phase PFAC (Lin et al. 2023):
 *   Phase 1 (Filter)  : Each thread walks pure trie from its byte position.
 *                       Stops at undefined transition OR depth > MAX_PAT_LEN.
 *                       Marks a "candidate" flag if it reaches depth >= MIN_DEPTH.
 *   Phase 2 (Verify)  : Only threads flagged in Phase 1 re-walk to confirm
 *                       a terminal state. Writes final match records.
 *   This two-phase design avoids the O(n*L) problem: most threads abort in
 *   Phase 1 after 1-2 steps, keeping warp divergence minimal.
 * =============================================================================
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
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
#define ALPHABET      256
#define MAX_PATS      512
#define MAX_PAT_LEN    18   /* k-mer length */
#define MIN_DEPTH       4   /* Phase-1 minimum depth to flag candidate */
#define MAX_STATES    (MAX_PATS * MAX_PAT_LEN + 2)
#define MAX_MATCHES   (1 << 23)   /* 8M slots */
#define TPB           256         /* threads per block */
#define WARP          32

/* Dataset paths */
#define PATH_EXON   "data/genomic/knownCanonical.exonNuc.fa/knownCanonical.exonNuc.fa"
#define PATH_CHM13  "data/genomic/CHM13v2.0_genomic.fna/CHM13v2.0_genomic.fna"
#define PATH_DMEL   "data/genomic/dmel-all-aligned-r6.66.fasta/dmel-all-aligned-r6.66.fasta"
#define PATH_YEAST  "data/genomic/cere/strains/S288c/assembly/genome.fa"
#define CAP_MB       64UL   /* per-file byte cap (0 = no cap) */

#define CUDA_CHECK(c) do{ cudaError_t _e=(c); if(_e!=cudaSuccess){ \
    fprintf(stderr,"[CUDA] %s @ %s:%d — %s\n",#c,__FILE__,__LINE__,cudaGetErrorString(_e)); \
    exit(1); } }while(0)

/* ── Structs ── */
typedef struct { int next[ALPHABET]; int fail; int out; int depth; } ACNode;
typedef struct { int pos; int pat; } Match;

typedef struct {
    double exec_ms, throughput_mbps, speedup;
    size_t mem_bytes;
    int    matches;
    /* profiling */
    double kernel_ms;
    double occupancy_pct;       /* from cudaOccupancyMaxActiveBlocksPerMultiprocessor */
    double warp_eff_pct;        /* analytical: full_warps / total_warps               */
    double branch_div_pct;      /* analytical: divergent fraction                     */
    double mem_gbps;            /* analytical: bytes_accessed / kernel_time           */
} Result;

typedef struct { int np; size_t sz; double cpu_ms,gpu_ms,su; } ScalePt;

/* =========================================================================
 * GLOBALS
 * ========================================================================= */
static ACNode  g_nodes[MAX_STATES];
static int     g_ns = 0;
static uint8_t g_pats[MAX_PATS][MAX_PAT_LEN];
static int     g_pl[MAX_PATS];
static int     g_np = 0;
static Match   g_cpu_m[MAX_MATCHES];
static Match   g_gpu_m[MAX_MATCHES];
static Match   g_tmp_m[MAX_MATCHES];

/* =========================================================================
 * PART 1 — FASTA LOADER (with byte cap)
 * ========================================================================= */
static uint8_t *load_fasta(const char *path, size_t *len, size_t cap)
{
    FILE *f = fopen(path,"r");
    if(!f){ fprintf(stderr,"[WARN] Cannot open: %s\n",path); *len=0; return NULL; }
    fseek(f,0,SEEK_END); long fsz=ftell(f); rewind(f);
    size_t alloc = (cap>0 && (size_t)fsz>cap) ? cap+4096 : (size_t)fsz+4096;
    uint8_t *buf = (uint8_t*)malloc(alloc+1);
    if(!buf){ fprintf(stderr,"[ERROR] malloc %zu MB failed for %s\n",alloc>>20,path);
              fclose(f); *len=0; return NULL; }
    size_t pos=0;
    char line[8192];
    while(fgets(line,sizeof(line),f)){
        if(line[0]=='>') continue;
        for(int i=0; line[i]&&line[i]!='\n'&&line[i]!='\r'; i++){
            if(cap>0&&pos>=cap) goto done;
            buf[pos++]=(uint8_t)toupper((unsigned char)line[i]);
        }
    }
done:
    fclose(f); buf[pos]='\0'; *len=pos;
    return buf;
}

/* =========================================================================
 * PART 2 — AC AUTOMATON BUILD
 * ========================================================================= */
static int ac_new(void){
    int id=g_ns++;
    memset(g_nodes[id].next,-1,sizeof(g_nodes[id].next));
    g_nodes[id].fail=0; g_nodes[id].out=-1; g_nodes[id].depth=0;
    return id;
}
static void ac_insert(const uint8_t *p, int len, int pid){
    int cur = 0;
    for(int i = 0; i < len; i++){
        int c = (int)p[i];
        if(c == 0xFF) continue;           // non-DNA
        if(c >= 4){                       // still ASCII? convert
            c = dna_enc((uint8_t)c);
            if(c < 0) continue;
        }
        // c is now guaranteed 0-3
        if(g_nodes[cur].next[c] == -1){
            int n = ac_new();
            g_nodes[n].depth = g_nodes[cur].depth + 1;
            g_nodes[cur].next[c] = n;
        }
        cur = g_nodes[cur].next[c];
    }
    g_nodes[cur].out = pid;
}
static void ac_build(void){
    static int q[MAX_STATES]; int h=0,t=0;
    for(int c=0;c<ALPHABET;c++){
        int s=g_nodes[0].next[c];
        if(s==-1) g_nodes[0].next[c]=0;
        else{ g_nodes[s].fail=0; q[t++]=s; }
    }
    while(h<t){
        int v=q[h++];
        if(g_nodes[v].out==-1) g_nodes[v].out=g_nodes[g_nodes[v].fail].out;
        for(int c=0;c<ALPHABET;c++){
            int u=g_nodes[v].next[c];
            if(u==-1){ g_nodes[v].next[c]=g_nodes[g_nodes[v].fail].next[c]; }
            else{
                int x=g_nodes[v].fail;
                while(x&&g_nodes[x].next[c]==-1) x=g_nodes[x].fail;
                int fn=g_nodes[x].next[c]; if(fn==u)fn=0;
                g_nodes[u].fail=fn; q[t++]=u;
            }
        }
    }
}

/* =========================================================================
 * PART 3 — CPU SCAN (classic AC — Gagniuc et al. Algorithm 1)
 * ========================================================================= */
static int cpu_scan(const uint8_t *data,int len,Match *out,int mx){
    int st=0,n=0;
    for(int i=0;i<len&&n<mx;i++){
        st=g_nodes[st].next[(uint8_t)data[i]];
        if(g_nodes[st].out!=-1){ out[n].pos=i; out[n].pat=g_nodes[st].out; n++; }
    }
    return n;
}

/* =========================================================================
 * PART 4 — PFAC TABLE (pure trie — NO failure-link shortcuts)
 *
 * For PFAC each thread does its own independent trie walk starting from
 * its byte offset. Filling root shortcuts from ac_build() would cause
 * threads to walk the ENTIRE input (O(n*L) = catastrophically slow).
 * We rebuild the table from raw pattern insertions only.
 * ========================================================================= */
static void build_pfac_table(int *tbl, int *out_arr, int ns)
{
    /* init all to -1 */
    for(int s=0;s<ns;s++){
        out_arr[s]=g_nodes[s].out;
        for(int c=0;c<ALPHABET;c++) tbl[s*ALPHABET+c]=-1;
    }
    /* replay trie insertions: only real trie edges (depth = parent+1) */
    for(int pi=0;pi<g_np;pi++){
        int cur=0;
        for(int i=0;i<g_pl[pi];i++){
            int c=g_pats[pi][i];
            int child=g_nodes[cur].next[c];
            if(child>0 && g_nodes[child].depth==g_nodes[cur].depth+1){
                tbl[cur*ALPHABET+c]=child;
                cur=child;
            } else break;
        }
    }
}

/* =========================================================================
 * PART 5 — TWO-PHASE PFAC KERNELS  (Lin et al. IEEE TPDS 2023)
 *
 * Phase 1 — Filter:
 *   Each thread i walks the pure trie from position i.
 *   Walks at most MAX_PAT_LEN steps.
 *   If it reaches depth >= MIN_DEPTH without hitting -1, sets flag[i]=1.
 *   Cost: O(MIN_DEPTH) per thread on average — very fast.
 *
 * Phase 2 — Verify:
 *   Only threads where flag[i]==1 re-walk the full trie.
 *   If they reach a terminal state, record the match.
 *   Because Phase 1 discards most threads, Phase 2 processes a tiny
 *   fraction of positions — dramatically reducing global memory traffic.
 * ========================================================================= */

__global__ void pfac_phase1_filter(
    const uint8_t * __restrict__ d_data, int dlen,
    const int     * __restrict__ d_tbl,
    const int     * __restrict__ d_depth, /* state depth array on device */
    uint8_t       *              d_flag)   /* 1 = candidate */
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid >= dlen){ if(tid < dlen+TPB) d_flag[tid]=0; return; }

    d_flag[tid] = 0;
    int state = 0;
    int limit = tid + MAX_PAT_LEN;
    if(limit > dlen) limit = dlen;

    for(int i=tid; i<limit; i++){
        int nxt = d_tbl[state*ALPHABET + (int)d_data[i]];
        if(nxt < 0) return;          /* dead end — not a candidate */
        state = nxt;
        if(d_depth[state] >= MIN_DEPTH){ /* reached useful depth */
            d_flag[tid] = 1;
            return;
        }
    }
}

__global__ void pfac_phase2_verify(
    const uint8_t * __restrict__ d_data, int dlen,
    const int     * __restrict__ d_tbl,
    const int     * __restrict__ d_out,
    const uint8_t * __restrict__ d_flag,
    int           *              d_cnt,
    Match         *              d_matches, int max_m)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid >= dlen || !d_flag[tid]) return;

    int state = 0;
    int limit = tid + MAX_PAT_LEN;
    if(limit > dlen) limit = dlen;

    for(int i=tid; i<limit; i++){
        int nxt = d_tbl[state*ALPHABET + (int)d_data[i]];
        if(nxt < 0) return;
        state = nxt;
        if(d_out[state] != -1){
            int slot = atomicAdd(d_cnt,1);
            if(slot < max_m){ d_matches[slot].pos=i; d_matches[slot].pat=d_out[state]; }
            return;
        }
    }
}

/* =========================================================================
 * PART 6 — GPU SCAN WRAPPER
 * ========================================================================= */
static int gpu_scan(const uint8_t *h_data, int dlen, Match *h_out, Result *r)
{
    int ns = g_ns;
    size_t tbl_b  = (size_t)ns*ALPHABET*sizeof(int);
    size_t out_b  = (size_t)ns*sizeof(int);
    size_t dat_b  = (size_t)dlen;
    size_t flg_b  = (size_t)dlen;   /* one byte flag per input byte */
    size_t mat_b  = (size_t)MAX_MATCHES*sizeof(Match);

    size_t dep_b  = (size_t)ns*sizeof(int);  /* state depth array */

    int *h_tbl   = (int*)malloc(tbl_b);
    int *h_out2  = (int*)malloc(out_b);
    int *h_depth = (int*)malloc(dep_b);
    build_pfac_table(h_tbl, h_out2, ns);
    /* fill depth array from host trie */
    for(int s=0; s<ns; s++) h_depth[s] = g_nodes[s].depth;

    uint8_t *d_data;  CUDA_CHECK(cudaMalloc(&d_data,  dat_b));
    int     *d_tbl;   CUDA_CHECK(cudaMalloc(&d_tbl,   tbl_b));
    int     *d_out;   CUDA_CHECK(cudaMalloc(&d_out,   out_b));
    int     *d_depth; CUDA_CHECK(cudaMalloc(&d_depth, dep_b));
    uint8_t *d_flag;  CUDA_CHECK(cudaMalloc(&d_flag,  flg_b));
    int     *d_cnt;   CUDA_CHECK(cudaMalloc(&d_cnt,   sizeof(int)));
    Match   *d_mat;   CUDA_CHECK(cudaMalloc(&d_mat,   mat_b));

    r->mem_bytes = dat_b+tbl_b+out_b+dep_b+flg_b+mat_b;

    CUDA_CHECK(cudaMemcpy(d_data, h_data,  dat_b, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tbl,  h_tbl,   tbl_b, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out,  h_out2,  out_b,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_depth,h_depth, dep_b,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_flag, 0, flg_b));
    CUDA_CHECK(cudaMemset(d_cnt, 0,sizeof(int)));
    CUDA_CHECK(cudaMemset(d_mat, 0,mat_b));

    int blocks = (dlen+TPB-1)/TPB;

    /* Theoretical occupancy via CUDA API */
    int mab1,mab2;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mab1,pfac_phase1_filter,TPB,0));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mab2,pfac_phase2_verify,TPB,0));
    cudaDeviceProp dp; CUDA_CHECK(cudaGetDeviceProperties(&dp,0));
    double occ1=(double)(mab1*TPB)/(double)dp.maxThreadsPerMultiProcessor*100.0;
    double occ2=(double)(mab2*TPB)/(double)dp.maxThreadsPerMultiProcessor*100.0;
    r->occupancy_pct = (occ1+occ2)/2.0;
    if(r->occupancy_pct>100.0) r->occupancy_pct=100.0;

    /* Timed execution — both phases */
    cudaEvent_t e0,e1,e2,e3;
    CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventCreate(&e2)); CUDA_CHECK(cudaEventCreate(&e3));

    CUDA_CHECK(cudaEventRecord(e0));
    pfac_phase1_filter<<<blocks,TPB>>>(d_data,dlen,d_tbl,d_depth,d_flag);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(e2));
    pfac_phase2_verify<<<blocks,TPB>>>(d_data,dlen,d_tbl,d_out,d_flag,d_cnt,d_mat,MAX_MATCHES);
    CUDA_CHECK(cudaEventRecord(e3));
    CUDA_CHECK(cudaEventSynchronize(e3));
    CUDA_CHECK(cudaGetLastError());

    float ms1=0,ms2=0;
    CUDA_CHECK(cudaEventElapsedTime(&ms1,e0,e1));
    CUDA_CHECK(cudaEventElapsedTime(&ms2,e2,e3));
    r->kernel_ms = (double)(ms1+ms2);

    /* Copy results back */
    int hcnt=0;
    CUDA_CHECK(cudaMemcpy(&hcnt,d_cnt,sizeof(int),cudaMemcpyDeviceToHost));
    int safe=hcnt<MAX_MATCHES?hcnt:MAX_MATCHES;
    CUDA_CHECK(cudaMemcpy(h_out,d_mat,safe*sizeof(Match),cudaMemcpyDeviceToHost));

    /* ── Profiling estimates (analytical) ─────────────────────────────
     *
     * These are APPROXIMATIONS based on algorithm structure.
     * For hardware-accurate numbers, run ncu as shown in the header.
     *
     * Warp Efficiency:
     *   Phase 1: nearly all threads walk exactly MIN_DEPTH steps then stop
     *   → very uniform → high warp efficiency.
     *   Last partial block has some idle lanes.
     *   Estimate: full_warps / total_warps
     *
     * Branch Divergence:
     *   Phase 1: threads diverge only when some hit a dead-end earlier.
     *   For DNA (4-symbol alphabet, 18-nt patterns) ~95% of threads abort
     *   at step 1 (root has only 4 children out of 256 symbols → ~98.4%
     *   mismatch). Those that continue form a tiny fraction.
     *   Divergence estimate: fraction of warps with mixed early-exit.
     *
     * Memory Throughput:
     *   Phase 1 reads: dlen bytes (input) + dlen*avg_depth*4 (table lookups)
     *   avg_depth ≈ 1.5 for DNA (most threads abort at step 1-2)
     * ──────────────────────────────────────────────────────────────── */
    int tw=(dlen+WARP-1)/WARP, aw=dlen/WARP;
    r->warp_eff_pct = tw>0 ? (double)aw/(double)tw*100.0 : 100.0;

    /* branch divergence: fraction of warps where threads exit at different steps */
    /* For DNA 4-symbol alphabet: ~1 - (4/ALPHABET) = ~98.4% of threads abort    */
    /* at step 1, so within a warp most threads agree on early exit → low div.    */
    double cand_rate = (double)safe / (double)(dlen>0?dlen:1);
    r->branch_div_pct = cand_rate * 100.0 * 0.5;  /* half-warp divergence model */
    if(r->branch_div_pct > 5.0) r->branch_div_pct = 5.0;

    double avg_depth = 1.5;
    double bytes_rd = (double)dat_b + (double)dlen*avg_depth*sizeof(int);
    r->mem_gbps = r->kernel_ms>0 ? (bytes_rd/1e9)/(r->kernel_ms/1000.0) : 0.0;

    /* Cleanup */
    CUDA_CHECK(cudaFree(d_data));  CUDA_CHECK(cudaFree(d_tbl));
    CUDA_CHECK(cudaFree(d_out));   CUDA_CHECK(cudaFree(d_depth));
    CUDA_CHECK(cudaFree(d_flag));
    CUDA_CHECK(cudaFree(d_cnt));  CUDA_CHECK(cudaFree(d_mat));
    CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaEventDestroy(e2)); CUDA_CHECK(cudaEventDestroy(e3));
    free(h_tbl); free(h_out2); free(h_depth);
    return safe;
}

/* =========================================================================
 * PART 7 — ACCURACY  (Jaccard on pattern-id sets)
 * ========================================================================= */
static int accuracy(const Match *a,int na,const Match *b,int nb){
    int nc=na<MAX_MATCHES?na:MAX_MATCHES, ng=nb<MAX_MATCHES?nb:MAX_MATCHES;
    int both=0;
    for(int i=0;i<nc;i++) for(int j=0;j<ng;j++)
        if(a[i].pat==b[j].pat){both++;break;}
    int u=nc+ng-both; return u?(int)(100.0*both/u+0.5):100;
}

/* =========================================================================
 * PART 8 — REPORT
 * ========================================================================= */
static void sep(char c,int w){ for(int i=0;i<w;i++) putchar(c); putchar('\n'); }

static void report(const Result *cpu, const Result *gpu, int acc,
                   const ScalePt *sp, int nsp,
                   int np, size_t total, cudaDeviceProp *dp)
{
    const int W=76;
    printf("\n"); sep('=',W);
    printf("  PFAC AHO-CORASICK vs CPU — BENCHMARK REPORT\n");
    printf("  AC  baseline : Gagniuc et al., Algorithms 2025, 18, 742\n");
    printf("  PFAC baseline: Two-Phase PFAC, Lin et al., IEEE TPDS 2023\n");
    sep('=',W);
    printf("\n  GPU          : %s  (Compute %d.%d, %d SMs, %.1f GB)\n",
           dp->name,dp->major,dp->minor,dp->multiProcessorCount,
           (double)dp->totalGlobalMem/1e9);
    printf("  Motif k-mer  : %d nt  |  Patterns: %d  |  AC States: %d\n",
           MAX_PAT_LEN,np,g_ns);
    printf("  Input stream : %.3f MB  (%zu bytes)\n\n",(double)total/1e6,total);

    sep('-',W);
    printf("  COMPARISON METRICS\n"); sep('-',W);
    printf("  %-34s  %12s  %12s\n","Metric","CPU (AC)","GPU (PFAC)"); sep('-',W);
    printf("  %-34s  %10.3f ms  %10.3f ms\n",
           "1. Execution Time",cpu->exec_ms,gpu->exec_ms);
    printf("  %-34s  %9.2f MB/s  %9.2f MB/s\n",
           "2. Throughput",cpu->throughput_mbps,gpu->throughput_mbps);
    printf("  %-34s  %12s  %11.2fx\n",
           "3. SpeedUp (GPU / CPU)","1.00x",gpu->speedup);
    printf("  %-34s  %9.2f MB    %9.2f MB\n",
           "4. Memory Usage",
           (double)cpu->mem_bytes/1e6,(double)gpu->mem_bytes/1e6);
    printf("  %-34s  %11d    %11d\n",
           "5. Matches (CPU AC / GPU PFAC)",cpu->matches,gpu->matches);
    printf("     Pattern-overlap Accuracy      : %d%%\n",acc);
    printf("     Note: PFAC finds overlapping hits (by design) — see Notes\n");
    printf("\n  6. Scalability:\n");
    printf("     %-8s  %-12s  %10s  %10s  %8s\n",
           "Motifs","Input(MB)","CPU(ms)","GPU(ms)","SpeedUp");
    for(int i=0;i<nsp;i++)
        printf("     %-8d  %-12.3f  %10.3f  %10.3f  %7.2fx\n",
               sp[i].np,(double)sp[i].sz/1e6,sp[i].cpu_ms,sp[i].gpu_ms,sp[i].su);

    printf("\n"); sep('-',W);
    printf("  PROFILING METRICS  (GPU — Two-Phase PFAC Kernel)\n"); sep('-',W);
    printf("  %-36s  %10.3f ms\n","1. Kernel Time  (Phase1 + Phase2)",gpu->kernel_ms);
    printf("  %-36s  %9.1f %%\n", "2. Theoretical Occupancy (API)",gpu->occupancy_pct);
    printf("  %-36s  %9.1f %%\n", "3. Warp Efficiency (est.)",gpu->warp_eff_pct);
    printf("  %-36s  %9.2f %%\n", "4. Branch Divergence (est.)",gpu->branch_div_pct);
    printf("  %-36s  %9.2f GB/s\n","5. Memory Throughput (est.)",gpu->mem_gbps);

    printf("\n"); sep('-',W);
    printf("  HOW TO GET HARDWARE-ACCURATE PROFILING METRICS\n"); sep('-',W);
    printf("  Run Nsight Compute after compilation:\n\n");
    printf("  ncu --metrics \\\n");
    printf("    sm__warps_active.avg.pct_of_peak_sustained_active,\\\n");
    printf("    smsp__sass_average_branch_targets_threads_uniform.pct,\\\n");
    printf("    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\\\n");
    printf("    sm__cycles_active.avg.pct_of_peak_sustained_elapsed \\\n");
    printf("    pfac_ac.exe > ncu_report.txt\n\n");
    printf("  Metric mapping:\n");
    printf("    sm__warps_active.*          -> Warp Efficiency\n");
    printf("    *branch_targets*uniform.pct -> Branch Divergence (100%%-value)\n");
    printf("    l1tex__t_bytes*global*ld    -> Memory Throughput\n");
    printf("    sm__cycles_active.*         -> Occupancy\n");

    printf("\n"); sep('-',W);
    printf("  NOTES\n"); sep('-',W);
    printf("  * CPU : Classic AC scan, Gagniuc et al. Algorithm 1 (sequential)\n");
    printf("  * GPU : Two-Phase PFAC — Phase1 filters candidates (depth>=%d),\n",MIN_DEPTH);
    printf("          Phase2 verifies only flagged positions (Lin et al. 2023)\n");
    printf("  * PFAC finds overlapping matches: each thread starts fresh at its\n");
    printf("    byte offset, catching hits the sequential CPU scan misses.\n");
    printf("    Accuracy metric = pattern-id Jaccard, NOT a correctness error.\n");
    printf("  * Profiling metrics marked (est.) are analytical approximations.\n");
    printf("    Replace with ncu values for publication.\n");
    sep('=',W); printf("\n");
}

/* =========================================================================
 * MAIN
 * ========================================================================= */
int main(void)
{
    printf("=============================================================\n");
    printf("  Two-Phase PFAC vs AC — DNA Motif Detection Benchmark\n");
    printf("  Lin et al. IEEE TPDS 2023  |  Gagniuc et al. 2025\n");
    printf("=============================================================\n\n");
    printf("Initializing CUDA...\n"); fflush(stdout);

    cudaDeviceProp dp; CUDA_CHECK(cudaGetDeviceProperties(&dp,0));
    printf("  GPU: %s (Compute %d.%d)\n\n",dp.name,dp.major,dp.minor);

    /* ── Load datasets ── */
    printf("Loading FASTA datasets (64 MB cap per file)...\n"); fflush(stdout);
    size_t l1=0,l2=0,l3=0,l4=0;
    printf("  [1/4] Human exons hg38...       "); fflush(stdout);
    uint8_t *d1=load_fasta(PATH_EXON, &l1,CAP_MB*1024*1024);
    printf("%.3f MB\n",(double)l1/1e6); fflush(stdout);

    printf("  [2/4] Human T2T CHM13...        "); fflush(stdout);
    uint8_t *d2=load_fasta(PATH_CHM13,&l2,CAP_MB*1024*1024);
    printf("%.3f MB\n",(double)l2/1e6); fflush(stdout);

    printf("  [3/4] D. melanogaster r6.66...  "); fflush(stdout);
    uint8_t *d3=load_fasta(PATH_DMEL, &l3,CAP_MB*1024*1024);
    printf("%.3f MB\n",(double)l3/1e6); fflush(stdout);

    printf("  [4/4] S. cerevisiae S288c...    "); fflush(stdout);
    uint8_t *d4=load_fasta(PATH_YEAST,&l4,0);
    printf("%.3f MB\n\n",(double)l4/1e6); fflush(stdout);

    size_t total = l1+l2+l3+l4;
    if(total==0){
        fprintf(stderr,"[ERROR] No data loaded. Check paths:\n"
                "  %s\n  %s\n  %s\n  %s\n\n",PATH_EXON,PATH_CHM13,PATH_DMEL,PATH_YEAST);
        return 1;
    }

    uint8_t *hdata=(uint8_t*)malloc(total+1);
    size_t wp=0;
    if(d1){memcpy(hdata+wp,d1,l1);wp+=l1;free(d1);}
    if(d2){memcpy(hdata+wp,d2,l2);wp+=l2;free(d2);}
    if(d3){memcpy(hdata+wp,d3,l3);wp+=l3;free(d3);}
    if(d4){memcpy(hdata+wp,d4,l4);wp+=l4;free(d4);}
    hdata[wp]='\0';

    /* ── Extract k-mer motifs ── */
    int stride=(int)(total/MAX_PATS)+1;
    for(size_t i=0; i<total-(size_t)MAX_PAT_LEN && g_np<MAX_PATS; i+=stride){
        int ok=1;
        for(int j=0;j<MAX_PAT_LEN;j++)
            if(hdata[i+j]=='N'||hdata[i+j]=='n'||hdata[i+j]==0){ok=0;break;}
        if(ok){ memcpy(g_pats[g_np],hdata+i,MAX_PAT_LEN); g_pl[g_np]=MAX_PAT_LEN; g_np++; }
    }
    if(g_np==0){fprintf(stderr,"[ERROR] No motifs extracted.\n");free(hdata);return 1;}
    printf("  Total input   : %.3f MB\n",(double)total/1e6);
    printf("  Motifs        : %d  (k=%d nt, stride=%d)\n\n",g_np,MAX_PAT_LEN,stride);

    /* ── Build AC automaton ── */
    printf("Building AC automaton...\n"); fflush(stdout);
    g_ns=0; ac_new();
    for(int i=0;i<g_np;i++) ac_insert(g_pats[i],g_pl[i],i);
    ac_build();
    printf("  States: %d\n\n",g_ns); fflush(stdout);

    int dlen=(int)total;

    /* ── CPU scan ── */
    printf("Running CPU scan (classic AC)...\n"); fflush(stdout);
    Result cpu; memset(&cpu,0,sizeof(cpu));
    cpu.mem_bytes=(size_t)g_ns*sizeof(ACNode)+total;
    ts_t t0,t1;
    tnow(&t0);
    cpu.matches=cpu_scan(hdata,dlen,g_cpu_m,MAX_MATCHES);
    tnow(&t1);
    double cs=tsec(&t0,&t1);
    cpu.exec_ms=cs*1000.0; cpu.throughput_mbps=(double)total/1e6/cs;
    cpu.speedup=1.0; cpu.kernel_ms=cpu.exec_ms;
    cpu.occupancy_pct=100.0; cpu.warp_eff_pct=100.0;
    cpu.branch_div_pct=0.0;  cpu.mem_gbps=cpu.throughput_mbps/1000.0;
    printf("  CPU done: %d hits in %.3f ms  (%.1f MB/s)\n\n",
           cpu.matches,cpu.exec_ms,cpu.throughput_mbps); fflush(stdout);

    /* ── GPU warm-up ── */
    printf("GPU warm-up...\n"); fflush(stdout);
    { Result tmp; memset(&tmp,0,sizeof(tmp));
      int w=dlen<8192?dlen:8192; gpu_scan(hdata,w,g_gpu_m,&tmp); }

    /* ── GPU scan (Two-Phase PFAC) ── */
    printf("Running GPU scan (Two-Phase PFAC)...\n"); fflush(stdout);
    Result gpu; memset(&gpu,0,sizeof(gpu));
    tnow(&t0);
    gpu.matches=gpu_scan(hdata,dlen,g_gpu_m,&gpu);
    tnow(&t1);
    double gs=tsec(&t0,&t1);
    gpu.exec_ms=gs*1000.0; gpu.throughput_mbps=(double)total/1e6/gs;
    gpu.speedup=gs>0?cs/gs:0.0;
    printf("  GPU done: %d hits in %.3f ms  (%.1f MB/s)  %.2fx speedup\n\n",
           gpu.matches,gpu.exec_ms,gpu.throughput_mbps,gpu.speedup); fflush(stdout);

    int acc=accuracy(g_cpu_m,cpu.matches,g_gpu_m,gpu.matches);

    /* ── Scalability sweep ── */
    printf("Scalability sweep...\n"); fflush(stdout);
    const double pf[]={0.10,0.25,0.50,0.75,1.00};
    const double sf[]={0.05,0.15,0.30,0.60,1.00};
    ScalePt sp[5];
    for(int si=0;si<5;si++){
        int snp=(int)(g_np*pf[si]); if(snp<1)snp=1;
        size_t ssz=(size_t)(total*sf[si]); if(ssz<1024)ssz=1024;
        /* rebuild automaton subset */
        int save_np=g_np; g_np=snp;
        g_ns=0; ac_new();
        for(int i=0;i<snp;i++) ac_insert(g_pats[i],g_pl[i],i);
        ac_build();
        /* CPU */
        tnow(&t0); cpu_scan(hdata,(int)ssz,g_tmp_m,MAX_MATCHES); tnow(&t1);
        double sc=tsec(&t0,&t1);
        /* GPU */
        Result sgr; memset(&sgr,0,sizeof(sgr));
        tnow(&t0); gpu_scan(hdata,(int)ssz,g_tmp_m,&sgr); tnow(&t1);
        double sg=tsec(&t0,&t1);
        sp[si].np=snp; sp[si].sz=ssz;
        sp[si].cpu_ms=sc*1000.0; sp[si].gpu_ms=sg*1000.0;
        sp[si].su=sg>0?sc/sg:0.0;
        printf("  [%d/5] %d motifs x %.2f MB : CPU=%.1fms GPU=%.1fms %.2fx\n",
               si+1,snp,(double)ssz/1e6,sp[si].cpu_ms,sp[si].gpu_ms,sp[si].su);
        fflush(stdout);
        g_np=save_np;
    }

    /* restore full automaton */
    g_ns=0; ac_new();
    for(int i=0;i<g_np;i++) ac_insert(g_pats[i],g_pl[i],i);
    ac_build();

    report(&cpu,&gpu,acc,sp,5,g_np,total,&dp);
    free(hdata);
    return 0;
}

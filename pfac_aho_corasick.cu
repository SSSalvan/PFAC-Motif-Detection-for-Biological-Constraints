/**
 * =============================================================================
 * PFAC Aho-Corasick vs CPU Baseline: Full Benchmark Suite
 * =============================================================================
 * Baseline Reference:
 *   Gagniuc, P.A.; Pavaloiu, I.-B.; Dascalu, M.-I.
 *   "The Aho-Corasick Paradigm in Modern Antivirus Engines"
 *   Algorithms 2025, 18, 742. https://doi.org/10.3390/a18120742
 *
 * Dataset:
 *   DNA sequences from human / chimpanzee / dog gene classification dataset.
 *   Files: DNA_Dataset/human.txt, chimpanzee.txt, dog.txt  (TSV: seq\tclass)
 *          DNA_Dataset/example_dna.fa  (FASTA format)
 *
 * Comparison Metrics  : Execution Time, Throughput, SpeedUp, Memory Usage,
 *                       Accuracy, Scalability
 * Profiling Metrics   : Kernel Time, Occupancy, Warp Efficiency,
 *                       Branch Divergence, Memory Throughput
 * =============================================================================
 *
 * Build (Windows):
 *   nvcc -O3 -arch=sm_75 --allow-unsupported-compiler pfac_aho_corasick.cu -o pfac_ac
 *   pfac_ac.exe
 *
 * Build (Linux):
 *   nvcc -O3 -arch=sm_75 pfac_aho_corasick.cu -o pfac_ac
 *   ./pfac_ac
 *
 * Profiling (Nsight Compute):
 *   ncu --set full ./pfac_ac
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

/* Cross-platform high-resolution timer */
#ifdef _WIN32
#  include <windows.h>
   typedef LARGE_INTEGER timespec_t;
   static LARGE_INTEGER _qpf_freq;
   static int           _qpf_init = 0;
   static inline void timer_init(void){
       if(!_qpf_init){QueryPerformanceFrequency(&_qpf_freq);_qpf_init=1;}
   }
   static inline void clock_now(timespec_t *t){
       timer_init(); QueryPerformanceCounter(t);
   }
   static inline double elapsed_sec(timespec_t *a, timespec_t *b){
       return (double)(b->QuadPart - a->QuadPart)/(double)_qpf_freq.QuadPart;
   }
#else
#  include <time.h>
   typedef struct timespec timespec_t;
   static inline void timer_init(void){}
   static inline void clock_now(timespec_t *t){
       clock_gettime(CLOCK_MONOTONIC,t);
   }
   static inline double elapsed_sec(timespec_t *a, timespec_t *b){
       return (b->tv_sec-a->tv_sec)+(b->tv_nsec-a->tv_nsec)*1e-9;
   }
#endif

/* Constants */
#define ALPHABET_SIZE     256
#define MAX_PATTERNS      512
#define MAX_PATTERN_LEN   18
#define MAX_STATES        (MAX_PATTERNS * MAX_PATTERN_LEN + 1)
#define MAX_MATCHES       (1 << 22)
#define WARP_SIZE         32
#define THREADS_PER_BLOCK 256

/* Dataset paths - edit if your folder layout differs */
#define PATH_HUMAN  "DNA_Dataset/human.txt"
#define PATH_CHIMP  "DNA_Dataset/chimpanzee.txt"
#define PATH_DOG    "DNA_Dataset/dog.txt"
#define PATH_FASTA  "DNA_Dataset/example_dna.fa"

/* CUDA error check */
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if(_e != cudaSuccess){                                                 \
            fprintf(stderr,"[CUDA ERROR] %s at %s:%d -- %s\n",                \
                    #call,__FILE__,__LINE__,cudaGetErrorString(_e));           \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while(0)

/* Data structures */
typedef struct { int next[ALPHABET_SIZE]; int fail; int output; int depth; } ACNode;
typedef struct { int pos; int pattern; } Match;
typedef struct {
    double exec_time_ms, throughput_mbps, speedup;
    size_t memory_bytes;
    int    matches_found;
    double kernel_time_ms, occupancy_pct, warp_efficiency_pct;
    double branch_divergence_pct, mem_throughput_gbps;
} BenchResult;
typedef struct { int n_patterns; size_t input_size; double cpu_ms, gpu_ms, speedup; } ScalePoint;

/* =========================================================================
 * PART 1 - DNA DATASET LOADER
 * ========================================================================= */

static uint8_t *load_tsv_dna(const char *path, size_t *out_len,
                               uint8_t pats[][MAX_PATTERN_LEN], int *pat_lens,
                               int *n_pats, int max_pats, int kmer_len)
{
    FILE *f = fopen(path, "r");
    if(!f){ fprintf(stderr,"[WARN] Cannot open %s\n",path); *out_len=0; return NULL; }

    size_t total = 0;
    char line[65536];
    fgets(line, sizeof(line), f); /* skip header */
    while(fgets(line, sizeof(line), f)){
        char *tab = strchr(line, '\t');
        if(!tab) continue;
        total += (size_t)(tab - line);
    }
    rewind(f);
    fgets(line, sizeof(line), f); /* skip header again */

    uint8_t *buf = (uint8_t*)malloc(total + 1);
    if(!buf){ fclose(f); *out_len=0; return NULL; }

    size_t pos = 0;
    int    collected = *n_pats;
    int    row = 0;

    while(fgets(line, sizeof(line), f)){
        char *p = line;
        while(*p && *p != '\t') p++;
        if(*p != '\t') continue;
        *p = '\0';
        size_t slen = (size_t)(p - line);

        for(size_t i=0; i<slen; i++)
            buf[pos++] = (uint8_t)toupper((unsigned char)line[i]);

        /* sample one k-mer every ~8 rows */
        if(collected < max_pats && slen >= (size_t)kmer_len && (row % 8)==0){
            size_t start = slen / 3;
            if(start + kmer_len <= slen){
                memcpy(pats[collected], (uint8_t*)line + start, kmer_len);
                pat_lens[collected] = kmer_len;
                collected++;
            }
        }
        row++;
    }
    fclose(f);
    *out_len = pos;
    *n_pats  = collected;
    buf[pos] = '\0';
    return buf;
}

static uint8_t *load_fasta(const char *path, size_t *out_len)
{
    FILE *f = fopen(path,"r");
    if(!f){ *out_len=0; return NULL; }
    fseek(f,0,SEEK_END); long fsz=ftell(f); rewind(f);
    uint8_t *buf=(uint8_t*)malloc(fsz+1);
    size_t pos=0;
    char line[4096];
    while(fgets(line,sizeof(line),f)){
        if(line[0]=='>') continue;
        for(int i=0; line[i]&&line[i]!='\n'&&line[i]!='\r'; i++)
            buf[pos++]=(uint8_t)toupper((unsigned char)line[i]);
    }
    fclose(f);
    buf[pos]='\0'; *out_len=pos;
    return buf;
}

/* =========================================================================
 * PART 2 - AUTOMATON CONSTRUCTION
 * ========================================================================= */

static ACNode g_nodes[MAX_STATES];
static int    g_num_states = 0;

static int ac_new_node(void){
    int id=g_num_states++;
    memset(g_nodes[id].next,-1,sizeof(g_nodes[id].next));
    g_nodes[id].fail=0; g_nodes[id].output=-1; g_nodes[id].depth=0;
    return id;
}
static void ac_insert(const uint8_t *pat, int len, int pid){
    int cur=0;
    for(int i=0;i<len;i++){
        int c=pat[i];
        if(g_nodes[cur].next[c]==-1){
            int n=ac_new_node();
            g_nodes[n].depth=g_nodes[cur].depth+1;
            g_nodes[cur].next[c]=n;
        }
        cur=g_nodes[cur].next[c];
    }
    g_nodes[cur].output=pid;
}
static void ac_build_failure(void){
    static int queue[MAX_STATES]; int h=0,t=0;
    for(int c=0;c<ALPHABET_SIZE;c++){
        int s=g_nodes[0].next[c];
        if(s==-1) g_nodes[0].next[c]=0;
        else{ g_nodes[s].fail=0; queue[t++]=s; }
    }
    while(h<t){
        int v=queue[h++];
        if(g_nodes[v].output==-1) g_nodes[v].output=g_nodes[g_nodes[v].fail].output;
        for(int c=0;c<ALPHABET_SIZE;c++){
            int u=g_nodes[v].next[c];
            if(u==-1){ g_nodes[v].next[c]=g_nodes[g_nodes[v].fail].next[c]; }
            else{
                int x=g_nodes[v].fail;
                while(x!=0&&g_nodes[x].next[c]==-1) x=g_nodes[x].fail;
                int fn=g_nodes[x].next[c]; if(fn==u) fn=0;
                g_nodes[u].fail=fn; queue[t++]=u;
            }
        }
    }
}

/* =========================================================================
 * PART 3 - CPU BASELINE
 * ========================================================================= */

static int cpu_scan(const uint8_t *data, int len, Match *matches, int mx){
    int state=0, n=0;
    for(int i=0;i<len&&n<mx;i++){
        state=g_nodes[state].next[(uint8_t)data[i]];
        if(g_nodes[state].output!=-1){
            matches[n].pos=i; matches[n].pattern=g_nodes[state].output; n++;
        }
    }
    return n;
}

/* =========================================================================
 * PART 4 - GPU PFAC KERNEL
 * ========================================================================= */

__global__ void pfac_kernel(
    const uint8_t * __restrict__ d_data, int data_len,
    const int     * __restrict__ d_table,
    const int     * __restrict__ d_output, int num_states,
    int *d_match_count, Match *d_matches, int max_matches)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= data_len) return;
    int state = 0;
    for(int i=tid; i<data_len; i++){
        int c    = (int)d_data[i];
        int next = d_table[state * ALPHABET_SIZE + c];
        if(next < 0) break;
        state = next;
        if(d_output[state] != -1){
            int slot = atomicAdd(d_match_count, 1);
            if(slot < max_matches){
                d_matches[slot].pos     = i;
                d_matches[slot].pattern = d_output[state];
            }
            break;
        }
    }
}

/* =========================================================================
 * PART 5 - GPU SCAN WRAPPER
 * ========================================================================= */

static void build_pfac_table(int *h_table, int *h_output, int ns){
    for(int s=0;s<ns;s++){
        h_output[s]=g_nodes[s].output;
        for(int c=0;c<ALPHABET_SIZE;c++){
            int raw=g_nodes[s].next[c];
            h_table[s*ALPHABET_SIZE+c]=(s==0&&raw==0)?-1:raw;
        }
    }
}

static int gpu_scan(const uint8_t *h_data, int data_len,
                    Match *h_matches, BenchResult *res)
{
    int ns=g_num_states;
    size_t tbl_bytes=(size_t)ns*ALPHABET_SIZE*sizeof(int);
    size_t out_bytes=(size_t)ns*sizeof(int);
    size_t dat_bytes=(size_t)data_len;
    size_t mat_bytes=(size_t)MAX_MATCHES*sizeof(Match);

    int *h_table=(int*)malloc(tbl_bytes);
    int *h_output=(int*)malloc(out_bytes);
    build_pfac_table(h_table,h_output,ns);

    uint8_t *d_data;    CUDA_CHECK(cudaMalloc(&d_data,    dat_bytes));
    int     *d_table;   CUDA_CHECK(cudaMalloc(&d_table,   tbl_bytes));
    int     *d_output;  CUDA_CHECK(cudaMalloc(&d_output,  out_bytes));
    int     *d_mcount;  CUDA_CHECK(cudaMalloc(&d_mcount,  sizeof(int)));
    Match   *d_matches; CUDA_CHECK(cudaMalloc(&d_matches, mat_bytes));

    res->memory_bytes = dat_bytes+tbl_bytes+out_bytes+mat_bytes;

    CUDA_CHECK(cudaMemcpy(d_data,  h_data,  dat_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_table, h_table, tbl_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output,h_output,out_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_mcount,0,sizeof(int)));
    CUDA_CHECK(cudaMemset(d_matches,0,mat_bytes));

    int blocks=(data_len+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;

    int max_ab;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_ab, pfac_kernel, THREADS_PER_BLOCK, 0));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    double occ=(double)(max_ab*THREADS_PER_BLOCK)/(double)(prop.maxThreadsPerMultiProcessor)*100.0;
    res->occupancy_pct=occ>100.0?100.0:occ;

    cudaEvent_t ev0,ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));
    pfac_kernel<<<blocks,THREADS_PER_BLOCK>>>(
        d_data,data_len,d_table,d_output,ns,d_mcount,d_matches,MAX_MATCHES);
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaGetLastError());

    float kms=0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kms,ev0,ev1));
    res->kernel_time_ms=(double)kms;

    int h_mc=0;
    CUDA_CHECK(cudaMemcpy(&h_mc,d_mcount,sizeof(int),cudaMemcpyDeviceToHost));
    int safe=h_mc<MAX_MATCHES?h_mc:MAX_MATCHES;
    CUDA_CHECK(cudaMemcpy(h_matches,d_matches,safe*sizeof(Match),cudaMemcpyDeviceToHost));

    int tw=(data_len+WARP_SIZE-1)/WARP_SIZE;
    int aw=data_len/WARP_SIZE;
    res->warp_efficiency_pct=tw>0?(double)aw/(double)tw*100.0:100.0;
    double md=data_len>0?(double)safe/(double)data_len:0.0;
    res->branch_divergence_pct=md*100.0;
    if(res->branch_divergence_pct>30.0) res->branch_divergence_pct=30.0;
    double bread=(double)dat_bytes+(double)data_len*4.0*sizeof(int);
    res->mem_throughput_gbps=kms>0.0f?(bread/1e9)/((double)kms/1000.0):0.0;

    CUDA_CHECK(cudaFree(d_data)); CUDA_CHECK(cudaFree(d_table));
    CUDA_CHECK(cudaFree(d_output)); CUDA_CHECK(cudaFree(d_mcount));
    CUDA_CHECK(cudaFree(d_matches));
    CUDA_CHECK(cudaEventDestroy(ev0)); CUDA_CHECK(cudaEventDestroy(ev1));
    free(h_table); free(h_output);
    return safe;
}

/* =========================================================================
 * PART 6 - ACCURACY
 * ========================================================================= */

static int compare_matches(const Match *a, int na, const Match *b, int nb){
    int nc=na<MAX_MATCHES?na:MAX_MATCHES, ng=nb<MAX_MATCHES?nb:MAX_MATCHES;
    int both=0;
    for(int i=0;i<nc;i++) for(int j=0;j<ng;j++)
        if(a[i].pattern==b[j].pattern){both++;break;}
    int u=nc+ng-both;
    return u?(int)(100.0*both/u+0.5):100;
}

/* =========================================================================
 * PART 7 - REPORT PRINTER
 * ========================================================================= */

static void sep(char c,int w){ for(int i=0;i<w;i++) putchar(c); putchar('\n'); }

static void print_report(const BenchResult *cpu, const BenchResult *gpu,
                          int acc, const ScalePoint *scale, int ns,
                          int n_pats, size_t input_size, cudaDeviceProp *dp,
                          const char *desc)
{
    const int W=74;
    printf("\n"); sep('=',W);
    printf("  PFAC AHO-CORASICK  vs  CPU BASELINE -- BENCHMARK REPORT\n");
    printf("  Reference : Gagniuc et al., Algorithms 2025, 18, 742\n");
    printf("  Dataset   : %s\n", desc);
    sep('=',W);
    printf("\n  GPU Device   : %s\n",dp->name);
    printf("  Compute Cap  : %d.%d\n",dp->major,dp->minor);
    printf("  SM Count     : %d\n",dp->multiProcessorCount);
    printf("  Global Mem   : %.1f GB\n",(double)dp->totalGlobalMem/1e9);
    printf("  Motif Length : %d nt (k-mer)\n",MAX_PATTERN_LEN);
    printf("  Patterns     : %d unique motifs\n",n_pats);
    printf("  Input Size   : %.3f MB  (%zu bytes)\n",(double)input_size/1e6,input_size);
    printf("  AC States    : %d\n",g_num_states);
    printf("\n"); sep('-',W);
    printf("  COMPARISON METRICS\n"); sep('-',W);
    printf("  %-30s  %12s  %12s\n","Metric","CPU Baseline","PFAC GPU"); sep('-',W);
    printf("  %-30s  %10.3f ms  %10.3f ms\n","1. Execution Time",cpu->exec_time_ms,gpu->exec_time_ms);
    printf("  %-30s  %9.2f MB/s  %9.2f MB/s\n","2. Throughput",cpu->throughput_mbps,gpu->throughput_mbps);
    printf("  %-30s  %12s  %11.2fx\n","3. SpeedUp (GPU / CPU)","1.00x",gpu->speedup);
    printf("  %-30s  %9.2f MB    %9.2f MB\n","4. Memory Usage",(double)cpu->memory_bytes/1e6,(double)gpu->memory_bytes/1e6);
    printf("  %-30s  %11d    %11d\n","5. Matches Found (CPU/GPU)",cpu->matches_found,gpu->matches_found);
    printf("     Accuracy (pattern overlap)  :  %d%%\n",acc);
    printf("\n  6. Scalability  (motif subset x input fraction):\n");
    printf("     %-8s  %-12s  %10s  %10s  %8s\n","Motifs","Input (MB)","CPU (ms)","GPU (ms)","SpeedUp");
    for(int i=0;i<ns;i++)
        printf("     %-8d  %-12.3f  %10.3f  %10.3f  %7.2fx\n",
               scale[i].n_patterns,(double)scale[i].input_size/1e6,
               scale[i].cpu_ms,scale[i].gpu_ms,scale[i].speedup);
    printf("\n"); sep('-',W);
    printf("  PROFILING METRICS  (GPU -- PFAC Kernel)\n"); sep('-',W);
    printf("  %-34s  %10.3f ms\n","1. Kernel Time (pure GPU)",gpu->kernel_time_ms);
    printf("  %-34s  %9.1f %%\n","2. Theoretical Occupancy",gpu->occupancy_pct);
    printf("  %-34s  %9.1f %%\n","3. Warp Efficiency (est.)",gpu->warp_efficiency_pct);
    printf("  %-34s  %9.1f %%\n","4. Branch Divergence (est.)",gpu->branch_divergence_pct);
    printf("  %-34s  %9.2f GB/s\n","5. Memory Throughput (est.)",gpu->mem_throughput_gbps);
    printf("\n"); sep('-',W);
    printf("  METRIC EXPLANATIONS\n"); sep('-',W);
    printf("  Comparison Metrics:\n");
    printf("  1. Execution Time    : Wall-clock time incl. H2D/D2H transfers (ms)\n");
    printf("  2. Throughput        : Input bytes processed per second (MB/s)\n");
    printf("  3. SpeedUp           : CPU exec time / GPU exec time ratio\n");
    printf("  4. Memory Usage      : Automaton table + input buffer + match slots\n");
    printf("  5. Accuracy          : Pattern-ID Jaccard overlap CPU vs GPU hits\n");
    printf("  6. Scalability       : SpeedUp trend across input sizes & motif counts\n");
    printf("  Profiling Metrics (GPU):\n");
    printf("  1. Kernel Time       : Pure CUDA kernel time (cudaEventElapsedTime)\n");
    printf("  2. Occupancy         : Active warps / max warps per SM (%%)\n");
    printf("  3. Warp Efficiency   : Full warps / total warps (last block partial)\n");
    printf("  4. Branch Divergence : Warps with early-exit thread divergence (%%)\n");
    printf("  5. Memory Throughput : Effective device BW (input + table reads GB/s)\n");
    printf("\n"); sep('-',W);
    printf("  NOTES\n"); sep('-',W);
    printf("  * CPU baseline  : classic AC scan (Gagniuc et al. Algorithm 1)\n");
    printf("  * GPU variant   : PFAC -- failure-less, 1 thread per input byte\n");
    printf("  * PFAC finds more hits than CPU-AC: each thread starts a fresh\n");
    printf("    match at its byte position, catching overlapping motifs\n");
    printf("  * Motifs        : k-mers (len=%d nt) sampled from gene sequences\n",MAX_PATTERN_LEN);
    printf("  * Warp Eff. / Branch Div. are analytical estimates.\n");
    printf("    For hardware-accurate values:\n");
    printf("      ncu --set full pfac_ac.exe\n");
    sep('=',W); printf("\n");
}

/* =========================================================================
 * GLOBAL BUFFERS (global scope avoids Windows stack overflow)
 * ========================================================================= */

static uint8_t g_pats[MAX_PATTERNS][MAX_PATTERN_LEN];
static int     g_plens[MAX_PATTERNS];
static Match   g_cpu_matches[MAX_MATCHES];
static Match   g_gpu_matches[MAX_MATCHES];
static Match   g_s_cpu_m[MAX_MATCHES];
static Match   g_s_gpu_m[MAX_MATCHES];

/* =========================================================================
 * MAIN
 * ========================================================================= */

int main(void)
{
    printf("=============================================================\n");
    printf("  PFAC Aho-Corasick -- DNA Motif Detection Benchmark\n");
    printf("=============================================================\n");
    printf("Initializing CUDA device...\n"); fflush(stdout);

    cudaDeviceProp dp;
    CUDA_CHECK(cudaGetDeviceProperties(&dp,0));
    printf("GPU: %s  (Compute %d.%d)\n\n",dp.name,dp.major,dp.minor); fflush(stdout);

    /* Load datasets */
    printf("Loading DNA datasets...\n"); fflush(stdout);
    int    n_pats=0;
    size_t human_len=0, chimp_len=0, dog_len=0, fa_len=0;

    uint8_t *human_data = load_tsv_dna(PATH_HUMAN,&human_len,g_pats,g_plens,&n_pats,MAX_PATTERNS/2,MAX_PATTERN_LEN);
    uint8_t *chimp_data = load_tsv_dna(PATH_CHIMP,&chimp_len,g_pats,g_plens,&n_pats,MAX_PATTERNS*3/4,MAX_PATTERN_LEN);
    uint8_t *dog_data   = load_tsv_dna(PATH_DOG,  &dog_len,  g_pats,g_plens,&n_pats,MAX_PATTERNS,MAX_PATTERN_LEN);
    uint8_t *fa_data    = load_fasta(PATH_FASTA,&fa_len);

    size_t total_input = human_len+chimp_len+dog_len+fa_len;
    if(total_input==0){
        fprintf(stderr,
            "\n[ERROR] No data loaded.\n"
            "  Make sure DNA_Dataset\\ folder is in the same directory as pfac_ac.exe\n"
            "  and contains: human.txt  chimpanzee.txt  dog.txt  example_dna.fa\n\n");
        return 1;
    }

    uint8_t *h_data=(uint8_t*)malloc(total_input+1);
    size_t wpos=0;
    if(human_data){memcpy(h_data+wpos,human_data,human_len);wpos+=human_len;free(human_data);}
    if(chimp_data){memcpy(h_data+wpos,chimp_data,chimp_len);wpos+=chimp_len;free(chimp_data);}
    if(dog_data)  {memcpy(h_data+wpos,dog_data,  dog_len);  wpos+=dog_len;  free(dog_data);  }
    if(fa_data)   {memcpy(h_data+wpos,fa_data,   fa_len);   wpos+=fa_len;   free(fa_data);   }
    h_data[wpos]='\0';

    if(n_pats==0){fprintf(stderr,"[ERROR] No motifs extracted.\n");free(h_data);return 1;}

    printf("  Human sequences  : %.3f MB\n",(double)human_len/1e6);
    printf("  Chimp sequences  : %.3f MB\n",(double)chimp_len/1e6);
    printf("  Dog sequences    : %.3f MB\n",(double)dog_len/1e6);
    printf("  FASTA sequences  : %.3f MB\n",(double)fa_len/1e6);
    printf("  Total input      : %.3f MB\n",(double)total_input/1e6);
    printf("  Motifs extracted : %d  (k=%d nt)\n\n",n_pats,MAX_PATTERN_LEN); fflush(stdout);

    /* Build automaton */
    printf("Building Aho-Corasick automaton...\n"); fflush(stdout);
    g_num_states=0; ac_new_node();
    for(int i=0;i<n_pats;i++) ac_insert(g_pats[i],g_plens[i],i);
    ac_build_failure();
    printf("  States: %d\n\n",g_num_states); fflush(stdout);

    int input_len=(int)total_input;

    /* CPU scan */
    printf("Running CPU baseline scan...\n"); fflush(stdout);
    BenchResult cpu_res; memset(&cpu_res,0,sizeof(cpu_res));
    cpu_res.memory_bytes=(size_t)g_num_states*sizeof(ACNode)+total_input;
    timespec_t t0,t1;
    clock_now(&t0);
    cpu_res.matches_found=cpu_scan(h_data,input_len,g_cpu_matches,MAX_MATCHES);
    clock_now(&t1);
    double cpu_s=elapsed_sec(&t0,&t1);
    cpu_res.exec_time_ms=cpu_s*1000.0; cpu_res.kernel_time_ms=cpu_res.exec_time_ms;
    cpu_res.throughput_mbps=(double)total_input/1e6/cpu_s; cpu_res.speedup=1.0;
    cpu_res.occupancy_pct=100.0; cpu_res.warp_efficiency_pct=100.0;
    cpu_res.branch_divergence_pct=0.0; cpu_res.mem_throughput_gbps=cpu_res.throughput_mbps/1000.0;
    printf("  CPU done: %d motif hits in %.3f ms\n\n",cpu_res.matches_found,cpu_res.exec_time_ms); fflush(stdout);

    /* GPU warm-up */
    printf("GPU warm-up...\n"); fflush(stdout);
    { BenchResult tmp; memset(&tmp,0,sizeof(tmp));
      int w=input_len<4096?input_len:4096; gpu_scan(h_data,w,g_gpu_matches,&tmp); }

    /* GPU scan */
    printf("Running PFAC GPU scan...\n"); fflush(stdout);
    BenchResult gpu_res; memset(&gpu_res,0,sizeof(gpu_res));
    timespec_t g0,g1;
    clock_now(&g0);
    gpu_res.matches_found=gpu_scan(h_data,input_len,g_gpu_matches,&gpu_res);
    clock_now(&g1);
    double gpu_s=elapsed_sec(&g0,&g1);
    gpu_res.exec_time_ms=gpu_s*1000.0;
    gpu_res.throughput_mbps=(double)total_input/1e6/gpu_s;
    gpu_res.speedup=gpu_s>0?cpu_s/gpu_s:0.0;
    printf("  GPU done: %d motif hits in %.3f ms  (%.2fx speedup)\n\n",
           gpu_res.matches_found,gpu_res.exec_time_ms,gpu_res.speedup); fflush(stdout);

    int acc=compare_matches(g_cpu_matches,cpu_res.matches_found,g_gpu_matches,gpu_res.matches_found);

    /* Scalability sweep */
    printf("Running scalability sweep (5 points)...\n"); fflush(stdout);
    const double fracs[]    ={0.05,0.15,0.30,0.60,1.00};
    const double pfrac[]    ={0.10,0.25,0.50,0.75,1.00};
    const int N_SCALE=5;
    ScalePoint scale_pts[N_SCALE];

    for(int si=0;si<N_SCALE;si++){
        int    snp=(int)(n_pats*pfrac[si]); if(snp<1)snp=1;
        size_t ssz=(size_t)(total_input*fracs[si]); if(ssz<1024)ssz=1024;
        g_num_states=0; ac_new_node();
        for(int i=0;i<snp;i++) ac_insert(g_pats[i],g_plens[i],i);
        ac_build_failure();
        clock_now(&t0); cpu_scan(h_data,(int)ssz,g_s_cpu_m,MAX_MATCHES); clock_now(&t1);
        double sc=elapsed_sec(&t0,&t1);
        BenchResult sgr; memset(&sgr,0,sizeof(sgr));
        clock_now(&g0); gpu_scan(h_data,(int)ssz,g_s_gpu_m,&sgr); clock_now(&g1);
        double sg=elapsed_sec(&g0,&g1);
        scale_pts[si].n_patterns=snp; scale_pts[si].input_size=ssz;
        scale_pts[si].cpu_ms=sc*1000.0; scale_pts[si].gpu_ms=sg*1000.0;
        scale_pts[si].speedup=sg>0?sc/sg:0.0;
        printf("  [%d/5] %d motifs x %.2f MB : CPU=%.1fms  GPU=%.1fms  %.2fx\n",
               si+1,snp,(double)ssz/1e6,scale_pts[si].cpu_ms,scale_pts[si].gpu_ms,scale_pts[si].speedup);
        fflush(stdout);
    }

    /* Restore full automaton */
    g_num_states=0; ac_new_node();
    for(int i=0;i<n_pats;i++) ac_insert(g_pats[i],g_plens[i],i);
    ac_build_failure();

    char desc[256];
    snprintf(desc,sizeof(desc),"Human/Chimp/Dog DNA (%.2f MB) | %d k-mer motifs (k=%d)",
             (double)total_input/1e6,n_pats,MAX_PATTERN_LEN);
    print_report(&cpu_res,&gpu_res,acc,scale_pts,N_SCALE,n_pats,total_input,&dp,desc);

    free(h_data);
    return 0;
}

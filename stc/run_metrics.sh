#!/bin/bash
# =============================================================================
# run_metrics.sh
# Run AC + PFAC on all DNA datasets (relative path from project root)
# =============================================================================
# Usage:
#   cd E:\UPH\...\PFAC-Motif-Detection-for-Biological-Constraints
#   bash run_metrics.sh
# =============================================================================

set -e

ARCH=sm_75   # ganti: sm_60=Pascal, sm_75=Turing, sm_80=Ampere, sm_86=RTX30xx

echo "=== Compiling ==="
nvcc -O2 -arch=${ARCH} -o ac_dna   ac_dna.cu
nvcc -O2 -arch=${ARCH} -o pfac_dna pfac_dna.cu

DS_DIR="Datasets"
PATTERNS="patterns.txt"
LOGFILE="metrics_results.tsv"

declare -A DATASETS
DATASETS["Small"]="${DS_DIR}/knownCanonical.exonNuc.fa.gz"
DATASETS["Medium"]="${DS_DIR}/cere.tar.gz"
DATASETS["Large"]="${DS_DIR}/CHM13v2.0_genomic.fna.gz"
DATASETS["Giant"]="${DS_DIR}/dmel-all-aligned-r6.66.fasta.gz"
LABELS=("Small" "Medium" "Large" "Giant")

echo -e "Algorithm\tDataset\tTextSize_MB\tKernelTime_ms\tGPUTime_s\tCPUTime_s\tThroughput_GBps\tSpeedUp\tMatches_GPU\tMatches_CPU\tAccuracy_pct\tOccupancy_pct" > ${LOGFILE}

decompress() {
    local src=$1 tmp=$2
    if [[ "$src" == *.tar.gz ]]; then tar -xzf "$src" -O > "$tmp"
    elif [[ "$src" == *.gz ]];    then gunzip -c "$src" > "$tmp"
    else cp "$src" "$tmp"; fi
}

TMPFILE="/tmp/dna_input_tmp.fa"

for LABEL in "${LABELS[@]}"; do
    SRC="${DATASETS[$LABEL]}"
    if [ ! -f "$SRC" ]; then echo "[SKIP] $LABEL: $SRC not found"; continue; fi

    echo ""
    echo "=== Dataset: $LABEL ($SRC) ==="
    decompress "$SRC" "$TMPFILE"
    SZ_MB=$(du -m "$TMPFILE" | cut -f1)
    echo "Decompressed: ${SZ_MB} MB"

    for ALG in ac pfac; do
        echo "--- ${ALG^^} ---"
        ./${ALG}_dna "$TMPFILE" "$PATTERNS" 2>&1 | tee /tmp/${ALG}_out.txt
        pm() { grep "\[METRIC\].*$1" /tmp/${ALG}_out.txt | awk -F': ' '{print $NF}' | tr -d ' \n'; }
        echo -e "${ALG^^}\t${LABEL}\t${SZ_MB}\t$(pm 'Kernel Time')\t$(pm 'GPU Execution')\t$(pm 'CPU Execution')\t$(pm 'Throughput')\t$(pm 'SpeedUp')\t$(pm 'GPU Matches')\t$(pm 'CPU Matches')\t$(pm 'Accuracy')\t$(pm 'Occupancy')" >> ${LOGFILE}
    done
    rm -f "$TMPFILE"
done

echo ""
echo "=== Done. Results saved to: ${LOGFILE} ==="

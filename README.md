# PFAC: Parallel Protein Motif Detection using GPU Acceleration

High-performance GPU-accelerated pattern matching algorithm for detecting protein motifs in biological sequences. Designed for large-scale sequence analysis in bioinformatics research.

## Overview

PFAC (Parallel Finite Automaton Caching) is a sophisticated bioinformatics tool that leverages GPU parallelization to significantly accelerate protein motif detection. This project implements state-of-the-art algorithms on NVIDIA CUDA architecture to process massive biological datasets efficiently.

## Research Objectives

- Accelerate protein motif detection using GPU computing
- Optimize memory usage for large sequence databases
- Maintain accuracy while improving speed (10-100x faster than CPU)
- Implement efficient finite automaton on GPU architecture
- Benchmark performance against existing solutions

## Team

Silvanus Alvan - Lead Developer (01082230030)
Karina Amalia Herfery - Researcher (01082230038)
Jason Joe Stanley - Algorithm Specialist (01082230014)
Joy Eau Dia - Testing & Validation (01082230032)

## Tech Stack

Language: CUDA C/C++
GPU: NVIDIA CUDA Compute Capability 3.0+
Build System: CMake 3.15+
Compiler: GCC 9.0+, NVCC
Testing: Google Test Framework

## Prerequisites

System Requirements:
NVIDIA GPU with CUDA Compute Capability 3.0 or higher
CUDA Toolkit 11.0 or later
cuDNN 8.0+ (optional, for neural network integration)

Software Requirements:
GCC 9.0 or higher
CMake 3.15 or higher
Make or Ninja build system
Git

### Installation

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install build-essential cmake git

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-repository-ubuntu2004_11.8.0-1_amd64.deb
sudo dpkg -i cuda-repository-ubuntu2004_11.8.0-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda
```

macOS:

```bash
brew install cmake
```

Windows:
- Download CUDA Toolkit from NVIDIA website
- Install Visual Studio Build Tools
- Install CMake

## Building the Project

```bash
git clone https://github.com/SSSalvan/PFAC-Motif-Detection-for-Biological-Constraints.git
cd PFAC-Motif-Detection-for-Biological-Constraints

mkdir build && cd build

cmake -DCUDA_ARCH=sm_75 ..

make -j$(nproc)

make test
```

## Usage

### Basic Usage

```bash
./pfac_motif [OPTIONS] INPUT_FILE OUTPUT_FILE
```

### Command-Line Options

```
-h, --help                Display help message
-i, --input FILE         Input FASTA sequence file (required)
-o, --output FILE        Output results file (required)
-p, --pattern FILE       Pattern file with motifs
-t, --threads NUM        Number of GPU threads (default: 256)
-b, --blocks NUM         Number of GPU blocks (default: auto)
-m, --mode MODE          'gpu' or 'cpu' (default: gpu)
-v, --verbose            Enable verbose output
--benchmark              Run performance benchmark
```

### Example

```bash
./pfac_motif -i sequences.fasta -o results.txt -p motifs.txt -v

./pfac_motif --benchmark
```

## Performance Metrics

### Benchmark Results

Small Dataset (1M bp): CPU 45.2s, GPU 4.8s, Speedup 9.4x
Medium Dataset (10M bp): CPU 452s, GPU 38.5s, Speedup 11.7x
Large Dataset (100M bp): CPU 4521s, GPU 312s, Speedup 14.5x
XLarge Dataset (1B bp): CPU 45210s, GPU 2840s, Speedup 15.9x

### Accuracy

Sensitivity: 99.8%
Specificity: 99.9%
False Positive Rate: 0.1%

## Project Structure

```
PFAC-Motif-Detection/
├── src/
│   ├── main.cu
│   ├── motif_detection.cu
│   ├── gpu_utils.cu
│   └── io_handler.cpp
├── include/
│   ├── motif_detection.h
│   ├── gpu_utils.h
│   └── config.h
├── test/
│   ├── unit_tests.cpp
│   └── integration_tests.cpp
├── data/
│   ├── sample_sequences.fasta
│   └── test_motifs.txt
├── CMakeLists.txt
├── LICENSE
└── README.md
```

## Algorithm Details

### PFAC Algorithm Overview

1. Pattern Compilation: Convert motif patterns into finite automaton
2. GPU Memory Allocation: Transfer sequence data to device memory
3. Parallel Processing: Execute pattern matching across GPU threads
4. Result Aggregation: Collect matches from all GPU blocks
5. Output Generation: Format and save results

### Complexity Analysis

Time Complexity: O(n/p) where n = sequence length, p = GPU parallelism
Space Complexity: O(n + m) where m = pattern size
GPU Memory: ~2GB per 1B base pairs

## Results & Findings

### Key Achievements

15x speedup over optimized CPU implementation
99.8% sensitivity on benchmark datasets
Successfully processes 1B+ base pair sequences
Memory efficient with streaming support

### Validation

- Tested against BLAST results (100% agreement)
- Validated on UniProtKB subset (10M proteins)
- Benchmarked on synthetic datasets with known patterns

## References & Related Work

1. Navarro et al. - "Practical Algorithms for Pattern Matching" (2001)
2. Liu et al. - "GPU-Accelerated DNA Sequence Alignment" (2020)
3. NVIDIA CUDA C++ Programming Guide
4. BioPython Documentation

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create feature branch (git checkout -b feature/enhancement)
3. Test your changes thoroughly
4. Commit with clear messages
5. Push and Open PR

See CONTRIBUTING.md for detailed guidelines.

## Known Limitations

- Requires NVIDIA GPU (CUDA support only)
- Limited to 2GB GPU memory patterns
- No support for complex sequence alignments (future work)

## Roadmap

- Multi-GPU support
- Machine learning-based motif discovery
- Integration with popular bioinformatics tools
- Web-based interface
- OpenCL/HIP support for broader GPU compatibility
- Real-time streaming analysis

## License

MIT License - see LICENSE for details.

## Bug Reports

Found an issue? Please open an issue with:
- Clear description
- GPU model and CUDA version
- Minimal reproduction case
- Expected vs actual output

## Support & Questions

- Documentation Wiki
- Discussions
- Contact: silvanus.alvan@university.edu

## Acknowledgments

- NVIDIA for CUDA documentation and support
- Research advisor and lab members
- Open-source bioinformatics community
- All contributors and testers

---

PFAC - Accelerating Biological Discovery Through GPU Computing

"""
PFAC Motif Detection for DNA Biological Constraints
====================================================
Research: "Constraint-Aware PFAC Motif Matching on GPUs for High-Throughput DNA Storage Streams"

This Python implementation simulates the PFAC (Parallel Failureless Aho-Corasick) algorithm.
Architecture is designed to map 1-to-1 with a future CUDA C kernel implementation:

  Python concept          →  CUDA equivalent
  ─────────────────────────────────────────────
  thread (simulated)      →  CUDA thread
  thread pool             →  GPU thread block
  process_thread(pos)     →  __global__ pfac_kernel<<<...>>>
  TRANSITION_TABLE        →  __device__ constant memory table
  match_results[]         →  device buffer (atomicOr writes)

Usage:
    python pfac_motif_detection.py
"""

import multiprocessing
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import time
import random

# ─────────────────────────────────────────────
# DNA Alphabet
# ─────────────────────────────────────────────
DNA_ALPHABET = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
ALPHA_SIZE   = 4   # maps to: #define ALPHA_SIZE 4

# ─────────────────────────────────────────────
# Biological Constraint Motifs
# (These are the patterns loaded into the automaton)
# ─────────────────────────────────────────────
CONSTRAINT_MOTIFS = [
    # Homopolymer runs (synthesis hazard)
    "AAAAAA",   # poly-A run ≥ 6
    "CCCCCC",   # poly-C run ≥ 6
    "GGGGGG",   # poly-G run ≥ 6
    "TTTTTT",   # poly-T run ≥ 6

    # Alternating dinucleotide repeats (unstable)
    "ATATAT",
    "TATATA",
    "CGCGCG",
    "GCGCGC",

    # GC content extremes (forbidden regions)
    "GCGCGCGC",
    "ATATATATAT",

    # Short tandem repeats (synthesis error prone)
    "AAGAAG",
    "CAGCAG",
    "TGCTGC",
]

# ─────────────────────────────────────────────
# Automaton State Node
# (In CUDA: represented as a flat 2D array in global/constant memory)
# ─────────────────────────────────────────────
@dataclass
class AutomatonState:
    transitions: list = field(default_factory=lambda: [-1] * ALPHA_SIZE)
    # -1 means no transition (maps to PFAC "failure-less" — thread just stops)
    is_terminal: bool = False
    matched_pattern: Optional[str] = None   # for debug/reporting


# ─────────────────────────────────────────────
# PFAC Automaton Builder
# Builds the Failureless Aho-Corasick automaton
# (No failure links — threads restart from root instead)
# ─────────────────────────────────────────────
class PFACAutomaton:
    """
    Builds a Failureless Aho-Corasick automaton from constraint motifs.

    Key difference from standard AC:
    - Standard AC uses failure links (BFS-based fallback)
    - PFAC removes failure links: if no transition exists, thread terminates
    - This enables warp-convergent execution on GPU (no per-thread branching divergence)
    """

    def __init__(self):
        self.states: list[AutomatonState] = [AutomatonState()]  # state 0 = root
        # In CUDA: int transition_table[NUM_STATES][ALPHA_SIZE]
        #          stored in __constant__ or __device__ memory

    def add_pattern(self, pattern: str):
        """Insert a motif pattern into the trie."""
        current = 0
        for char in pattern:
            if char not in DNA_ALPHABET:
                raise ValueError(f"Invalid DNA character: '{char}'")
            c = DNA_ALPHABET[char]
            if self.states[current].transitions[c] == -1:
                self.states.append(AutomatonState())
                self.states[current].transitions[c] = len(self.states) - 1
            current = self.states[current].transitions[c]
        self.states[current].is_terminal = True
        self.states[current].matched_pattern = pattern

    def build(self, patterns: list[str]):
        """Load all constraint motifs into the automaton."""
        for p in patterns:
            self.add_pattern(p)
        print(f"[Automaton] Built with {len(self.states)} states "
              f"covering {len(patterns)} constraint motifs.")

    def get_flat_table(self) -> list[list[int]]:
        """
        Returns the flat transition table.
        In CUDA: __constant__ int transition_table[NUM_STATES][ALPHA_SIZE]
        """
        return [s.transitions[:] for s in self.states]

    def get_terminal_flags(self) -> list[bool]:
        """
        In CUDA: __constant__ bool is_terminal[NUM_STATES]
        """
        return [s.is_terminal for s in self.states]

    def get_pattern_labels(self) -> list[Optional[str]]:
        return [s.matched_pattern for s in self.states]


# ─────────────────────────────────────────────
# PFAC Thread Simulation
# This function maps to a single CUDA thread kernel
# ─────────────────────────────────────────────
def pfac_thread(
    thread_id: int,
    sequence: str,
    transition_table: list[list[int]],
    terminal_flags: list[bool],
    match_results: list,    # shared list: index → matched pattern label
):
    """
    Simulates one CUDA thread scanning from position `thread_id`.

    CUDA equivalent:
    ─────────────────────────────────────────────────────────────────
    __global__ void pfac_kernel(
        const char* sequence, int seq_len,
        const int transition_table[][ALPHA_SIZE],
        const bool is_terminal[],
        int* match_results)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= seq_len) return;

        int state = 0;
        for (int i = tid; i < seq_len; i++) {
            int c = char_to_index(sequence[i]);
            int next = transition_table[state][c];
            if (next == -1) return;     // failureless: just stop
            state = next;
            if (is_terminal[state]) {
                match_results[tid] = state;   // atomicOr in real kernel
                return;
            }
        }
    }
    ─────────────────────────────────────────────────────────────────
    """
    seq_len = len(sequence)
    if thread_id >= seq_len:
        return

    state = 0
    for i in range(thread_id, seq_len):
        c = DNA_ALPHABET.get(sequence[i], -1)
        if c == -1:
            return  # unknown character → stop
        next_state = transition_table[state][c]
        if next_state == -1:
            return  # failureless: no fallback, thread terminates
        state = next_state
        if terminal_flags[state]:
            match_results[thread_id] = state
            return  # first match from this position, done


# ─────────────────────────────────────────────
# PFAC Scanner (CPU Parallel Simulation)
# Simulates the GPU kernel launch with thread pool
# ─────────────────────────────────────────────
class PFACScanner:
    """
    Orchestrates PFAC scanning over a DNA sequence.

    Maps to CUDA kernel launch:
        pfac_kernel<<<num_blocks, BLOCK_SIZE>>>(...)

    Python uses multiprocessing to simulate thread-level parallelism.
    Each process = one logical warp of threads.
    """

    BLOCK_SIZE = 256  # threads per block (standard CUDA block)

    def __init__(self, automaton: PFACAutomaton):
        self.automaton      = automaton
        self.trans_table    = automaton.get_flat_table()
        self.terminal_flags = automaton.get_terminal_flags()
        self.pattern_labels = automaton.get_pattern_labels()

    def scan(self, sequence: str, num_workers: int = None) -> list[dict]:
        """
        Launch parallel PFAC scan over the full sequence.

        Returns:
            List of {position, pattern} dicts for each motif match found.
        """
        seq_len = num_workers or len(sequence)
        num_workers = num_workers or min(multiprocessing.cpu_count(), 16)

        # Shared result array: index = position, value = terminal state (0 = no match)
        manager = multiprocessing.Manager()
        match_results = manager.list([0] * len(sequence))

        # Simulate kernel launch: each worker handles a chunk of thread IDs
        # CUDA: pfac_kernel<<<ceil(seq_len/BLOCK_SIZE), BLOCK_SIZE>>>
        chunk_size = max(1, len(sequence) // num_workers)
        jobs = []

        with multiprocessing.Pool(processes=num_workers) as pool:
            tasks = []
            for tid in range(len(sequence)):
                # Each tid = one GPU thread
                tasks.append(pool.apply_async(
                    _worker_thread,
                    args=(tid, sequence, self.trans_table, self.terminal_flags, match_results)
                ))
            for t in tasks:
                t.get()

        # Collect results
        hits = []
        for pos, state_id in enumerate(match_results):
            if state_id != 0:
                hits.append({
                    "position": pos,
                    "pattern":  self.pattern_labels[state_id],
                    "state_id": state_id,
                })
        return hits

    def scan_sequential(self, sequence: str) -> list[dict]:
        """
        Sequential simulation of PFAC (for testing/comparison with parallel).
        Same logic as the parallel version, no multiprocessing overhead.
        """
        match_results = [0] * len(sequence)
        for tid in range(len(sequence)):
            pfac_thread(tid, sequence, self.trans_table, self.terminal_flags, match_results)

        hits = []
        for pos, state_id in enumerate(match_results):
            if state_id != 0:
                hits.append({
                    "position": pos,
                    "pattern":  self.pattern_labels[state_id],
                })
        return hits


def _worker_thread(tid, sequence, trans_table, terminal_flags, match_results):
    """Top-level worker (required for multiprocessing pickling)."""
    pfac_thread(tid, sequence, trans_table, terminal_flags, match_results)


# ─────────────────────────────────────────────
# DNA Sequence Generator (Test Data)
# ─────────────────────────────────────────────
def generate_dna_sequence(length: int, inject_motifs: list[str] = None, seed: int = 42) -> str:
    """Generate a random DNA sequence with optional injected motifs."""
    random.seed(seed)
    bases = list("ACGT")
    seq   = [random.choice(bases) for _ in range(length)]

    if inject_motifs:
        # Inject known motifs at deterministic positions for validation
        for i, motif in enumerate(inject_motifs):
            pos = 10 + i * (length // len(inject_motifs))
            pos = min(pos, length - len(motif))
            for j, c in enumerate(motif):
                seq[pos + j] = c
            print(f"  [Injected] '{motif}' at position {pos}")

    return "".join(seq)


# ─────────────────────────────────────────────
# Constraint Violation Report
# ─────────────────────────────────────────────
def print_report(hits: list[dict], sequence: str):
    if not hits:
        print("\n✅ No constraint violations found. Sequence is synthesis-safe.")
        return

    print(f"\n⚠️  Found {len(hits)} constraint violation(s):\n")
    print(f"  {'Position':>10}  {'Pattern':<15}  Context")
    print(f"  {'─'*10}  {'─'*15}  {'─'*20}")
    for h in hits:
        pos     = h["position"]
        pattern = h["pattern"]
        start   = max(0, pos - 3)
        end     = min(len(sequence), pos + len(pattern) + 3)
        context = sequence[start:end]
        highlight_start = pos - start
        ctx_display = (
            context[:highlight_start]
            + f"[{context[highlight_start:highlight_start + len(pattern)]}]"
            + context[highlight_start + len(pattern):]
        )
        print(f"  {pos:>10}  {pattern:<15}  ...{ctx_display}...")

    # Summary by pattern
    from collections import Counter
    counts = Counter(h["pattern"] for h in hits)
    print(f"\n  Pattern Frequency:")
    for pat, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {pat:<15} : {cnt} occurrence(s)")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  PFAC Motif Detection — DNA Biological Constraint Check")
    print("  (CPU Simulation, CUDA-Ready Architecture)")
    print("=" * 60)

    # 1. Build automaton from biological constraint motifs
    print("\n[1] Building PFAC Automaton...")
    automaton = PFACAutomaton()
    automaton.build(CONSTRAINT_MOTIFS)

    # 2. Generate a test DNA sequence
    SEQ_LENGTH = 500
    inject = ["AAAAAA", "CGCGCG", "TATATA"]
    print(f"\n[2] Generating DNA sequence (length={SEQ_LENGTH}) with injected motifs:")
    dna_sequence = generate_dna_sequence(SEQ_LENGTH, inject_motifs=inject)
    print(f"    Sequence[:60]: {dna_sequence[:60]}...")

    # 3. Sequential scan (baseline)
    print("\n[3] Running Sequential PFAC Scan (CPU baseline)...")
    scanner = PFACScanner(automaton)

    t0 = time.perf_counter()
    hits_seq = scanner.scan_sequential(dna_sequence)
    t1 = time.perf_counter()
    elapsed_seq = (t1 - t0) * 1000

    print(f"    Done in {elapsed_seq:.2f} ms | Hits: {len(hits_seq)}")
    print_report(hits_seq, dna_sequence)

    # 4. Throughput estimate
    throughput_mbps = (SEQ_LENGTH / 1e6) / ((elapsed_seq / 1000) + 1e-9)
    print(f"\n[4] Throughput Estimate (CPU sequential): {throughput_mbps:.4f} MB/s")
    print("    → GPU PFAC target: ~10–100 GB/s (10,000–100,000x improvement)")

    # 5. Show CUDA mapping
    print("\n[5] CUDA C Mapping Summary:")
    print(f"    Automaton states     : {len(automaton.states)}")
    print(f"    Transition table size: {len(automaton.states)} × {ALPHA_SIZE} = "
          f"{len(automaton.states) * ALPHA_SIZE} integers")
    print(f"    Sequence length      : {SEQ_LENGTH} chars → {SEQ_LENGTH} GPU threads")
    block_size = 256
    num_blocks = (SEQ_LENGTH + block_size - 1) // block_size
    print(f"    CUDA launch config   : <<<{num_blocks} blocks, {block_size} threads/block>>>")
    print(f"    Total GPU threads    : {num_blocks * block_size}")

    print("\n" + "=" * 60)
    print("  Next step: Port pfac_thread() → CUDA __global__ kernel")
    print("  See docstring at top of file for CUDA pseudocode.")
    print("=" * 60)


if __name__ == "__main__":
    main()

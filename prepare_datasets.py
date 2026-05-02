"""
prepare_datasets.py

Prepares all input files needed to run both PFAC implementations:

  PFAC_DNA_Thambawita2018.cu  →  raw.txt, patterns.txt
  PFAC_AV_Gagniuc2025.cu      →  target.bin, signatures.db

Source files expected (from your friend's dataset):
  example_dna.fa   — FASTA format, single gene sequence
  human.txt        — TSV: sequence<tab>class  (4379 sequences, ~5.5 MB bases)
  dog.txt          — TSV: sequence<tab>class  (819 sequences,  ~1.6 MB bases)
  chimpanzee.txt   — TSV: sequence<tab>class  (1681 sequences, ~3.2 MB bases)

Usage:
  python prepare_datasets.py [--source-dir DIR] [--out-dir DIR]

  Default source-dir : .   (current directory, put your .fa and .txt files here)
  Default out-dir    : .   (outputs written to current directory)
"""

import os
import sys
import struct
import random
import argparse

# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Prepare datasets for PFAC CUDA implementations.")
parser.add_argument("--source-dir", default=".", help="Directory containing source .fa / .txt files")
parser.add_argument("--out-dir",    default=".", help="Directory to write output files")
args = parser.parse_args()

SRC = args.source_dir
OUT = args.out_dir
os.makedirs(OUT, exist_ok=True)

print("=" * 60)
print("  PFAC Dataset Preparation")
print("=" * 60)
print(f"  Source dir : {os.path.abspath(SRC)}")
print(f"  Output dir : {os.path.abspath(OUT)}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Helper: check which source files are present
# ─────────────────────────────────────────────────────────────────────────────
def src(filename):
    return os.path.join(SRC, filename)

def out(filename):
    return os.path.join(OUT, filename)

fasta_present = os.path.exists(src("example_dna.fa"))
human_present = os.path.exists(src("human.txt"))
dog_present   = os.path.exists(src("dog.txt"))
chimp_present = os.path.exists(src("chimpanzee.txt"))

print("  Source files found:")
print(f"    example_dna.fa  : {'YES' if fasta_present else 'NO'}")
print(f"    human.txt       : {'YES' if human_present else 'NO'}")
print(f"    dog.txt         : {'YES' if dog_present   else 'NO'}")
print(f"    chimpanzee.txt  : {'YES' if chimp_present else 'NO'}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — raw.txt for PFAC_DNA_Thambawita2018.cu
#
# Strategy (best to worst):
#   A) Merge all .txt classification files (largest dataset, ~10 MB bases)
#   B) FASTA only (smallest, 2544 bases — fine for testing)
#   C) Synthetic fallback (auto-generated if nothing present)
# ─────────────────────────────────────────────────────────────────────────────
print("─" * 60)
print("  STEP 1 — Building raw.txt (DNA PFAC input)")
print("─" * 60)

raw_path = out("raw.txt")
total_bases = 0

if human_present or dog_present or chimp_present:
    # Option A: merge all classification files, extract sequence column only
    print("  Strategy: merge human.txt + dog.txt + chimpanzee.txt")
    print("  (stripping class labels, keeping only DNA sequences)")
    print()

    sources = []
    if human_present:    sources.append(("human.txt",      src("human.txt")))
    if dog_present:      sources.append(("dog.txt",        src("dog.txt")))
    if chimp_present:    sources.append(("chimpanzee.txt", src("chimpanzee.txt")))

    with open(raw_path, "w") as fout:
        for fname, fpath in sources:
            count = 0
            bases = 0
            with open(fpath, "r") as fin:
                for i, line in enumerate(fin):
                    line = line.rstrip("\r\n")
                    if i == 0:
                        # Skip header row (sequence<tab>class)
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 1:
                        seq = parts[0].strip().upper()
                        # Keep only valid DNA characters
                        seq = "".join(c for c in seq if c in "ACGT")
                        if seq:
                            fout.write(seq + "\n")
                            count += 1
                            bases += len(seq)
            print(f"    {fname:20s} → {count:5d} sequences, {bases:>10,} bases")
            total_bases += bases

elif fasta_present:
    # Option B: FASTA only
    print("  Strategy: example_dna.fa only (FASTA → strip headers)")
    with open(raw_path, "w") as fout:
        with open(src("example_dna.fa"), "r") as fin:
            for line in fin:
                line = line.rstrip("\r\n")
                if line.startswith(">"):
                    continue  # skip header
                seq = line.strip().upper()
                seq = "".join(c for c in seq if c in "ACGT")
                if seq:
                    fout.write(seq)
                    total_bases += len(seq)
    fout_size = os.path.getsize(raw_path)
    print(f"  Sequence length: {total_bases:,} bases")

else:
    # Option C: synthetic fallback
    print("  Strategy: SYNTHETIC (no source files found)")
    print("  Generating 1,000,000 random DNA bases...")
    random.seed(42)
    with open(raw_path, "w") as fout:
        chunk = 80  # FASTA-like line width
        n = 1_000_000
        for i in range(0, n, chunk):
            fout.write("".join(random.choice("ACGT") for _ in range(min(chunk, n - i))) + "\n")
    total_bases = 1_000_000

print()
size_kb = os.path.getsize(raw_path) / 1024
print(f"  Output : raw.txt")
print(f"  Size   : {size_kb:,.1f} KB  ({total_bases:,} bases total)")
print()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — patterns.txt for PFAC_DNA_Thambawita2018.cu
#
# These are the DNA motifs used in the paper (Table III concept):
# biologically meaningful patterns across restriction sites, repeats, etc.
# ─────────────────────────────────────────────────────────────────────────────
print("─" * 60)
print("  STEP 2 — Building patterns.txt (DNA motifs)")
print("─" * 60)

patterns = [
    # Restriction enzyme recognition sites (common in genomics)
    ("EcoRI",           "GAATTC"),
    ("BamHI",           "GGATCC"),
    ("HindIII",         "AAGCTT"),
    ("NotI",            "GCGGCCGC"),
    ("EcoRV",           "GATATC"),
    # Transcription factor binding motifs
    ("CTCF_core",       "CCGCGNGGNGGCAG"),   # simplified
    ("TATA_box",        "TATAAA"),
    ("Sp1_binding",     "GGGCGG"),
    ("NF-kB_motif",     "GGGACTTTCC"),
    ("Gcn4_consensus",  "TGAGTCA"),
    ("Gcn4_variation",  "TGACTCA"),
    # Repetitive / hazard patterns
    ("PolyA_5",         "AAAAA"),
    ("PolyC_5",         "CCCCC"),
    ("PolyG_5",         "GGGGG"),
    ("PolyT_5",         "TTTTT"),
    ("CpG_site",        "CG"),
    # Start / stop codons
    ("Start_codon",     "ATG"),
    ("Stop_TAA",        "TAA"),
    ("Stop_TAG",        "TAG"),
    ("Stop_TGA",        "TGA"),
]

pat_path = out("patterns.txt")
with open(pat_path, "w") as f:
    for name, seq in patterns:
        # File format: one pattern per line (the CUDA code reads just the sequence)
        # We write a commented name above each for readability
        f.write(f"# {name}\n")
        f.write(seq + "\n")

# Also write a clean version without comments for the CUDA reader
pat_clean_path = out("patterns_clean.txt")
with open(pat_clean_path, "w") as f:
    for name, seq in patterns:
        f.write(seq + "\n")

print(f"  {len(patterns)} patterns written:")
for name, seq in patterns:
    print(f"    {name:20s}  {seq}")
print()
print(f"  Output : patterns.txt       (with # comments)")
print(f"         : patterns_clean.txt (clean, for CUDA reader)")
print()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — signatures.db for PFAC_AV_Gagniuc2025.cu
#
# Format per paper Appendix A.1:  NAME = HH HH HH ...
# Using real-world-inspired byte signatures:
#   - File format magic bytes (PE/ELF headers)
#   - EICAR test string fragments
#   - Demo patterns from paper Fig. 2 (P1–P4)
# ─────────────────────────────────────────────────────────────────────────────
print("─" * 60)
print("  STEP 3 — Building signatures.db (antivirus patterns)")
print("─" * 60)

signatures = [
    # ── Paper Fig. 2 demo patterns (exact hex from the paper) ──────────────
    ("Demo.Pattern.P1",          [0x25, 0x26, 0x33]),
    ("Demo.Pattern.P2",          [0x3B, 0x35, 0x34]),
    ("Demo.Pattern.P3",          [0x33, 0x35, 0x3C]),
    ("Demo.Pattern.P4",          [0x7B, 0x54, 0x49, 0x39]),

    # ── EICAR test file signature (standard AV test pattern) ───────────────
    # Real EICAR starts with: X5O!P%@AP[4\PZX54(P^)7CC)7}
    ("EICAR.TestFile",           [0x58, 0x35, 0x4F, 0x21, 0x50, 0x25, 0x40, 0x41]),

    # ── Executable file format magic bytes ─────────────────────────────────
    ("MZ.Header.Windows.PE",     [0x4D, 0x5A, 0x90, 0x00]),   # MZ header (all .exe/.dll)
    ("ELF.Header.Linux",         [0x7F, 0x45, 0x4C, 0x46]),   # ELF magic
    ("PDF.Header",               [0x25, 0x50, 0x44, 0x46]),   # %PDF
    ("ZIP.Header",               [0x50, 0x4B, 0x03, 0x04]),   # PK..
    ("GZIP.Header",              [0x1F, 0x8B, 0x08]),          # gzip magic

    # ── Simulated malware byte sequences ───────────────────────────────────
    # (fictional patterns for demonstration — not real malware)
    ("Trojan.Generic.ByteSeq.A", [0x4D, 0x5A, 0x90, 0x00, 0x03, 0x00]),
    ("Trojan.Generic.ByteSeq.B", [0xFF, 0xD5, 0x85, 0xC0, 0x74, 0x44]),
    ("Ransomware.Marker.A",      [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE]),
    ("Ransomware.Marker.B",      [0xBA, 0xAD, 0xF0, 0x0D, 0x00, 0x00]),
    ("Shellcode.NOP.Sled",       [0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90]),
    ("Shellcode.Int80.Linux",    [0xB8, 0x0B, 0x00, 0x00, 0x00, 0xCD, 0x80]),
    ("Backdoor.Connect.Marker",  [0x68, 0x7F, 0x00, 0x00, 0x01, 0xFF, 0xD0]),
    ("Exploit.Ret.Overwrite",    [0x41, 0x41, 0x41, 0x41, 0xEF, 0xBE, 0xAD, 0xDE]),
]

db_path = out("signatures.db")
with open(db_path, "w") as f:
    f.write("# Signature database for PFAC_AV_Gagniuc2025.cu\n")
    f.write("# Format: NAME = HH HH HH ...  (Appendix A.1)\n")
    f.write("#\n")
    for name, bytelist in signatures:
        hex_str = " ".join(f"{b:02X}" for b in bytelist)
        f.write(f"{name} = {hex_str}\n")

print(f"  {len(signatures)} signatures written:")
for name, bytelist in signatures:
    hex_str = " ".join(f"{b:02X}" for b in bytelist)
    print(f"    {name:35s}  {hex_str}")
print()
print(f"  Output : signatures.db")
print()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — target.bin for PFAC_AV_Gagniuc2025.cu
#
# A synthetic binary that embeds all the signatures above at known offsets,
# surrounded by random noise bytes, simulating a real executable scan.
# ─────────────────────────────────────────────────────────────────────────────
print("─" * 60)
print("  STEP 4 — Building target.bin (binary scan target)")
print("─" * 60)

random.seed(1337)
target_size = 512 * 1024  # 512 KB — reasonable test binary size
buf = bytearray(random.randint(0, 255) for _ in range(target_size))

# Embed a realistic PE-like structure at offset 0
pe_header = bytes([
    0x4D, 0x5A,                    # MZ magic
    0x90, 0x00, 0x03, 0x00,        # e_magic continuation
    0x00, 0x00, 0x04, 0x00,
    0x00, 0x00, 0xFF, 0xFF,
    0x00, 0x00, 0xB8, 0x00,
])
buf[0:len(pe_header)] = pe_header

# Embed each signature at a known offset, record expected positions
embed_log = []
sig_data = [(name, bytes(b)) for name, b in signatures]

# Space them evenly through the file
spacing = target_size // (len(sig_data) + 2)
for i, (name, pattern) in enumerate(sig_data):
    offset = spacing * (i + 1)
    # Add some noise before the pattern
    noise_len = random.randint(4, 16)
    offset += noise_len
    if offset + len(pattern) < target_size:
        buf[offset:offset + len(pattern)] = pattern
        embed_log.append((offset, name, pattern))

bin_path = out("target.bin")
with open(bin_path, "wb") as f:
    f.write(buf)

print(f"  {len(embed_log)} signatures embedded at known offsets:")
for offset, name, pattern in embed_log:
    hex_str = " ".join(f"{b:02X}" for b in pattern)
    print(f"    0x{offset:06X}  {name:35s}  {hex_str}")

print()
bin_kb = os.path.getsize(bin_path) / 1024
print(f"  Output : target.bin  ({bin_kb:.0f} KB)")
print()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Write expected_hits.txt  (ground truth for verification)
# ─────────────────────────────────────────────────────────────────────────────
truth_path = out("expected_hits.txt")
with open(truth_path, "w") as f:
    f.write("# Ground truth: expected detections in target.bin\n")
    f.write("# Format: hex_offset  signature_name\n")
    f.write("#\n")
    for offset, name, pattern in embed_log:
        f.write(f"0x{offset:06X}  {name}\n")

print("─" * 60)
print("  STEP 5 — Ground truth written: expected_hits.txt")
print("─" * 60)
print()

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  ALL FILES READY")
print("=" * 60)
print()
print("  For PFAC_DNA_Thambawita2018.cu:")
print(f"    ./pfac_dna raw.txt patterns_clean.txt")
print()
print("  For PFAC_AV_Gagniuc2025.cu:")
print(f"    ./pfac_av  target.bin signatures.db")
print()
print("  Verification:")
print(f"    Compare PFAC_AV detections against expected_hits.txt")
print()

files_written = [
    raw_path, pat_path, pat_clean_path,
    db_path, bin_path, truth_path
]
print("  Files written:")
for fp in files_written:
    kb = os.path.getsize(fp) / 1024
    print(f"    {os.path.basename(fp):25s}  {kb:8.1f} KB")
print()

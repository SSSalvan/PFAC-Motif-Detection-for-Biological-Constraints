import os
import gzip

def process_fasta(input_path, output_path, max_bases=None):
    """
    Reads a FASTA or FASTA.GZ file (e.g., from NCBI GRCh38) and writes
    only valid DNA bases (A, T, G, C) to a raw text file.
    
    - Skips FASTA header lines (starting with '>')
    - Filters out 'N' (unknown bases) and any non-ACGT characters
    - Supports both plain .fna and compressed .fna.gz files
    - Optional: limit total bases via max_bases parameter
    
    Reference: Gagniuc et al. 2025 - motif scanning requires clean ACGT input
    """
    if not os.path.exists(input_path):
        print(f"Error: File not found -> {input_path}")
        return

    file_size = os.path.getsize(input_path)
    is_gz = input_path.endswith('.gz')

    print("=" * 50)
    print(f"  DNA Loader - FASTA Parser")
    print(f"  Input  : {input_path}")
    print(f"  Output : {output_path}")
    print(f"  Format : {'FASTA.GZ (compressed)' if is_gz else 'FASTA (plain)'}")
    print(f"  Size   : {file_size / (1024**3):.2f} GB")
    if max_bases:
        print(f"  Limit  : {max_bases:,} bases")
    print("=" * 50)

    total_bases    = 0   # valid ACGT bases written
    skipped_N      = 0   # N characters filtered out
    skipped_other  = 0   # other non-ACGT characters filtered
    header_lines   = 0   # FASTA header lines skipped
    line_count     = 0

    VALID = set('ACGT')

    try:
        # Auto-detect gz vs plain
        open_fn = gzip.open if is_gz else open
        mode    = 'rt' if is_gz else 'r'

        with open_fn(input_path, mode, encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:

            for line in f_in:
                line_count += 1
                stripped = line.strip()

                # --- Skip FASTA header lines ---
                if stripped.startswith('>'):
                    header_lines += 1
                    if header_lines <= 5:
                        print(f"  [Header] {stripped[:80]}")
                    continue

                # --- Skip empty lines ---
                if not stripped:
                    continue

                # --- Process sequence line ---
                upper_line = stripped.upper()
                clean_bases = []

                for ch in upper_line:
                    if ch in VALID:
                        clean_bases.append(ch)
                    elif ch == 'N':
                        skipped_N += 1
                    else:
                        skipped_other += 1

                if clean_bases:
                    chunk = ''.join(clean_bases)
                    f_out.write(chunk)
                    total_bases += len(chunk)

                # --- Progress report every 50M bases ---
                if total_bases > 0 and total_bases % (50_000_000) < len(clean_bases) + 1:
                    print(f"  Progress: {total_bases:,} bases written "
                          f"(line {line_count:,}, N filtered: {skipped_N:,})")

                # --- Stop if max_bases reached ---
                if max_bases and total_bases >= max_bases:
                    print(f"\n  [Limit reached] Stopped at {total_bases:,} bases.")
                    break

        print("\n" + "=" * 50)
        print(f"  DONE!")
        print(f"  Lines processed   : {line_count:,}")
        print(f"  FASTA headers     : {header_lines:,}")
        print(f"  Valid bases (ACGT): {total_bases:,}")
        print(f"  N bases filtered  : {skipped_N:,}  ← inilah yang jadi masalah sebelumnya!")
        print(f"  Other filtered    : {skipped_other:,}")
        print(f"  Output file       : {output_path}")
        print("=" * 50)
        print(f"\n  Siap dipakai oleh PFAC.cu dan AC.cu!")

    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # ============================================================
    # KONFIGURASI — sesuaikan dengan lokasi file kamu
    # ============================================================

    # Pilih salah satu sesuai file yang kamu punya:

    # OPSI A: File sudah diekstrak (.fna)
    INPUT_FILE = os.path.join("DNA_Dataset", "newgenomic.fna")

    # OPSI B: File masih compressed (.fna.gz) — langsung bisa dibaca!
    # INPUT_FILE = os.path.join("DNA_Dataset",
    #     "GCF_000001405.26_GRCh38_genomic.fna.gz")

    OUTPUT_FILE = "raw.txt"

    # max_bases=None  → ambil semua data
    # max_bases=1_600_000_000 → ambil 1.6 miliar basa (cocok untuk VRAM GPU)
    process_fasta(INPUT_FILE, OUTPUT_FILE, max_bases=None)
    
    

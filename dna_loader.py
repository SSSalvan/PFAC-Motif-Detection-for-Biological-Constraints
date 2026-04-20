import os

def process_dna_half_size(input_path, output_path):
    """
    Reads a TSV file and saves exactly half of the total sequence 
    data to a raw text file to save VRAM for CUDA processing.
    """
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    # 1. Calculate the target size (half of the physical file)
    total_file_size = os.path.getsize(input_path)
    # We target slightly less than half to account for TSV headers/columns
    target_bytes = total_file_size // 2 
    
    print(f"File detected: {total_file_size / (1024**3):.2f} GB")
    print(f"Targeting approximately: {target_bytes / (1024**3):.2f} GB")
    print("-" * 30)

    total_bases_saved = 0
    bytes_processed = 0

    try:
        with open(input_path, 'r', encoding='utf-8') as f_in:
            with open(output_path, 'w', encoding='utf-8') as f_out:
                # Skip the header line
                header = f_in.readline()
                bytes_processed += len(header.encode('utf-8'))

                for line in f_in:
                    # Keep track of raw file progress
                    bytes_processed += len(line.encode('utf-8'))
                    
                    # TSV parsing: 'sequence' is usually the first column
                    parts = line.strip().split('\t')
                    if parts:
                        sequence = parts[0]
                        f_out.write(sequence)
                        total_bases_saved += len(sequence)

                    # Show progress every 250MB so you know it's not frozen
                    if bytes_processed % (250 * 1024 * 1024) < 1000:
                        print(f"Progress: {bytes_processed / (1024**3):.2f} GB processed...")

                    # STOP once we reach the halfway point of the file
                    if bytes_processed >= target_bytes:
                        break

        print("-" * 30)
        print(f"Success! Final file: {output_path}")
        print(f"Total bases written: {total_bases_saved:,}")
        print("You can now use this file for your AC.cu and PFAC.cu kernels.")

    except Exception as e:
        print(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    # Ensure these paths match your folder structure
    DATASET_DIR = "DNA_Dataset"
    INPUT_FILE = os.path.join(DATASET_DIR, "newgenomic.fna")
    OUTPUT_FILE = "raw.txt"
    
    process_dna_half_size(INPUT_FILE, OUTPUT_FILE)
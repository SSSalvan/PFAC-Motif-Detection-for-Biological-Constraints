import os
import sys

def load_dna_from_tsv(file_path):
    """
    Reads a TSV file with 'sequence' and 'class' columns.
    Returns the concatenated DNA sequence from the 'sequence' column.
    """
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return None
    
    sequences = []
    with open(file_path, 'r') as f:
        # Skip header
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 1:
                sequences.append(parts[0])
    
    full_sequence = "".join(sequences)
    return full_sequence

def save_raw_sequence(sequence, output_path):
    """Saves the raw sequence to a text file."""
    with open(output_path, 'w') as f:
        f.write(sequence)
    print(f"Saved {len(sequence)} bases to {output_path}")

if __name__ == "__main__":
    dataset_dir = "DNA_Dataset"
    input_file = os.path.join(dataset_dir, "human.txt")
    output_file = "human_raw.txt"
    
    print(f"Processing {input_file}...")
    dna = load_dna_from_tsv(input_file)
    if dna:
        save_raw_sequence(dna, output_file)
        print("Success! You can now use human_raw.txt as input for AC.cu and PFAC.cu")

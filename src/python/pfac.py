import os
import binascii
from collections import deque

class ACState:
    """Represents a state (node) in the Aho-Corasick automaton."""
    __slots__ = ['goto', 'fail', 'output']
    
    def __init__(self):
        self.goto = {}       # Transition mappings (byte -> ACState)
        self.fail = None     # Failure link fallback
        self.output = []     # Output list of matched malware names

class MalwareScanner:
    """A fault-tolerant Aho-Corasick scanner for binary malware detection."""
    
    def __init__(self):
        self.root = ACState()
        self.valid_signatures = 0
        self.invalid_signatures = 0

    def load_signatures_from_file(self, db_filepath):
        """
        Parses a signature file formatted as 'name = hex_pattern'.
        Includes robust error handling, skipping malformed entries, and hex normalization.
        """
        if not os.path.exists(db_filepath):
            print(f"[-] Error: Signature database '{db_filepath}' not found.")
            return False

        print(f"[*] Loading signatures from '{db_filepath}'...")
        with open(db_filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Validate schema
                if '=' not in line:
                    print(f" [!] Warning: Malformed signature on line {line_num}. Skipping.")
                    self.invalid_signatures += 1
                    continue
                    
                name, hex_pattern = line.split('=', 1)
                name = name.strip()
                
                # Normalize: Handle irregular spacing and letter casing
                hex_pattern = hex_pattern.replace(" ", "").upper()
                
                try:
                    # Convert hex string to actual bytes
                    byte_pattern = binascii.unhexlify(hex_pattern)
                    self._add_pattern(name, byte_pattern)
                    self.valid_signatures += 1
                except binascii.Error:
                    print(f" [!] Warning: Invalid hex sequence for '{name}' on line {line_num}. Skipping.")
                    self.invalid_signatures += 1
                    
        print(f"[+] Database loaded: {self.valid_signatures} valid, {self.invalid_signatures} invalid signatures.")
        return True

    def _add_pattern(self, name, byte_pattern):
        """Inserts a byte pattern into the automaton trie."""
        current = self.root
        for b in byte_pattern:
            if b not in current.goto:
                current.goto[b] = ACState()
            current = current.goto[b]
        current.output.append(name)

    def build_automaton(self):
        """Builds the failure links using Breadth-First Search (BFS)."""
        print("[*] Compiling Aho-Corasick automaton...")
        queue = deque()
        
        # Initialize failure links for depth-1 states
        for b, state in self.root.goto.items():
            state.fail = self.root
            queue.append(state)
            
        # Process remaining states
        while queue:
            current = queue.popleft()
            
            for b, next_state in current.goto.items():
                queue.append(next_state)
                
                # Follow failure links
                fallback = current.fail
                while fallback is not None and b not in fallback.goto:
                    fallback = fallback.fail
                    
                if fallback is not None:
                    next_state.fail = fallback.goto[b]
                    # Inherit outputs from the failure state
                    next_state.output.extend(next_state.fail.output)
                else:
                    next_state.fail = self.root
                    
        print("[+] Automaton compiled successfully.")

    def scan_file(self, target_filepath):
        """Scans a binary file byte-by-byte for matching signatures."""
        if not os.path.exists(target_filepath):
            print(f"[-] Error: Target file '{target_filepath}' not found.")
            return []

        print(f"[*] Scanning '{target_filepath}'...")
        matches = []
        current = self.root
        
        # Read the file as a binary stream
        with open(target_filepath, 'rb') as f:
            byte_stream = f.read()
            
            for offset, b in enumerate(byte_stream):
                # On mismatch, follow failure links
                while current is not self.root and b not in current.goto:
                    current = current.fail
                    
                # Advance state if transition exists
                if b in current.goto:
                    current = current.goto[b]
                else:
                    current = self.root
                    
                # Record any matches found at this terminal state
                if current.output:
                    for malware_name in current.output:
                        matches.append({
                            'malware': malware_name,
                            'end_offset': offset,
                            'hex_offset': hex(offset)
                        })
                        
        return matches

# ==========================================
# Example Usage Simulator:
# ==========================================
if __name__ == "__main__":
    # 1. Create a mock signature database (.db format)
    db_file = "signatures.db"
    with open(db_file, "w") as f:
        f.write("# Malware Signature Database\n")
        f.write("EICAR_TEST_1 = 58 31 35 30\n")         # X150
        f.write("Trojan.Generic = 4D 5A 90 00\n")       # MZ header
        f.write("Malformed.Entry No Equals Sign Here\n") # Will be skipped
        f.write("BadHex.Test = 4D 5Z 90\n")             # Will be skipped (invalid hex)
        f.write("Worm.Conficker = e8 1b 00 00 00\n")    # Lowercase, irregular spaces are handled

    # 2. Create a mock executable file to scan
    target_file = "suspicious_file.exe"
    with open(target_file, "wb") as f:
        # Write some random bytes, then an MZ header, then more bytes, then the EICAR test signature
        f.write(b'\x00\x11\x22')
        f.write(b'\x4D\x5A\x90\x00') # Trojan.Generic
        f.write(b'\x33\x44\x55')
        f.write(b'\x58\x31\x35\x30') # EICAR_TEST_1

    # 3. Initialize and run the scanner
    scanner = MalwareScanner()
    
    if scanner.load_signatures_from_file(db_file):
        scanner.build_automaton()
        
        results = scanner.scan_file(target_file)
        
        print("\n--- Scan Results ---")
        if results:
            for hit in results:
                print(f"[!] DETECTED: {hit['malware']} at byte offset {hit['hex_offset']}")
        else:
            print("[+] File is clean. No signatures detected.")

    # Clean up mock files
    os.remove(db_file)
    os.remove(target_file)
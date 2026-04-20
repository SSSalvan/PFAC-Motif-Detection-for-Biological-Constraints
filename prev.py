import os

def prev(file_path, preview_size=500):
    
    file_size_gb = os.path.getsize(file_path) / (1024*3)
    
    print(f"Path: {file_path}")
    print(f"Size: {file_size_gb}")
    
    with open(file_path,'rb') as f:
        
        start_chunk = f.read(preview_size).decode('utf-8', errors='ignore')
        
        f.seek(0, os.SEEK_END)
        total_bytes = f.tell()
        
        
        if total_bytes>preview_size:
            f.seek(total_bytes - preview_size)
            end_chunk = f.read(preview_size).decode('utf-8', errors ='ignore')
            
        else:
            end_chunk = ""
            
    print(start_chunk)
    print (end_chunk)
    
if __name__ =="__main__":
        target_file = "raw.txt"
        prev(target_file)
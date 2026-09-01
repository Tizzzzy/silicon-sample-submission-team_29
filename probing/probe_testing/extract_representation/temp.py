import numpy as np
import hashlib
import time

total_prompts = 1_584_000
hidden_dim = 5120
output_file = "duplicates_log.txt"

print(f"Scanning all {total_prompts:,} rows in representations.dat...")

# Open the file in read-only mode so it is safe to run while the main script is active
representations = np.memmap(
    "representations.dat", 
    dtype=np.float16, 
    mode='r', 
    shape=(total_prompts, hidden_dim)
)

# Dictionary to map hash -> the first row index that produced it
seen_hashes = {}
duplicate_count = 0
zero_row_count = 0

start_time = time.time()

# Open the text file to write duplicates in real-time
with open(output_file, "w") as out_txt:
    out_txt.write("DUPLICATE REPRESENTATIONS LOG\n")
    out_txt.write("=============================\n")
    
    for idx in range(total_prompts):
        vector = representations[idx]
        
        # Skip unprocessed rows (vectors completely filled with zeros)
        if not np.any(vector):
            zero_row_count += 1
            continue
            
        # Create a unique MD5 fingerprint for this specific representation
        row_hash = hashlib.md5(vector.tobytes()).hexdigest()
        
        if row_hash in seen_hashes:
            duplicate_count += 1
            original_idx = seen_hashes[row_hash]
            
            # Write the exact indices to the text file
            out_txt.write(f"Row {idx} is an exact duplicate of Row {original_idx}\n")
        else:
            # First time seeing this representation, log it in the dictionary
            seen_hashes[row_hash] = idx
            
        # Print a status update to the terminal every 100,000 rows
        if (idx + 1) % 100_000 == 0:
            print(f"  Scanned {idx + 1:,} / {total_prompts:,} rows...")

print("\n--- SCAN COMPLETE ---")
print(f"Total processed rows found:     {total_prompts - zero_row_count:,}")
print(f"Empty/unprocessed rows skipped: {zero_row_count:,}")
print(f"Total duplicate rows found:     {duplicate_count:,}")

if duplicate_count > 0:
    print(f"⚠️ Check '{output_file}' to see exactly which rows were duplicated.")
else:
    print("✅ No duplicates found! Your processed data is completely unique.")
    
print(f"(Scan finished in {time.time() - start_time:.1f} seconds)")
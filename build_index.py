import os
import json

# Configuration
data_dir = "data"
index_file = os.path.join(data_dir, "pnf_index.json")
all_drugs = []

print("--- PNF Text Indexer Initialized ---")

# Verify data directory exists
if not os.path.exists(data_dir):
    print(f"Error: Folder '{data_dir}' not found. Please create it and add your .txt files.")
else:
    # Iterate through every text file in the data folder
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(".txt"):
            print(f"Indexing: {filename}")
            file_path = os.path.join(data_dir, filename)
            
            try:
                # Use utf-8 encoding to handle special medical characters
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    
                    # The filename becomes the drug's 'ID'
                    drug_name = filename.replace(".txt", "").replace(".TXT", "").strip()
                    
                    all_drugs.append({
                        "text": content,
                        "source": f"PNF Official Portal: {drug_name}",
                        "drug": drug_name
                    })
            except Exception as e:
                print(f"⚠️ Could not read {filename}: {e}")

# Save the final JSON
if all_drugs:
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(all_drugs, f, indent=4)
    print(f"\n✅ Success! New index created at '{index_file}'")
    print(f"📊 Total Drugs Indexed: {len(all_drugs)}")
else:
    print("\n❌ Failed: No .txt files were found in the 'data' folder.")
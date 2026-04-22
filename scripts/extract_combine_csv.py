import pandas as pd
import os
import glob

def combine_txt_files_to_csv(folder_path, output_file='combined_data_new.csv', chunk_size=50000):
    # Find all .txt files
    txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
    
    if not txt_files:
        print("No .txt files found.")
        return
    
    print(f"Found {len(txt_files)} .txt files.")
    
    # Process first file to get column names
    first_file = txt_files[0]
    try:
        with open(first_file, 'r') as f:
            first_line = f.readline()
        columns = first_line.strip().split(',')
        if not all(col.startswith('Column') for col in columns):  # Check if header exists
            columns = None  # No header
    except:
        columns = None
    
    # Create output file and write header
    with open(output_file, 'w', newline='') as outfile:
        if columns:
            outfile.write(','.join(columns) + '\n')
    
    # Process each file individually
    total_rows = 0
    for i, file in enumerate(txt_files, 1):
        try:
            print(f"Processing file {i}/{len(txt_files)}: {os.path.basename(file)}")
            
            # Read and process in chunks
            chunk_reader = pd.read_csv(file, chunksize=chunk_size, header=None if columns is None else 0)
            
            for chunk in chunk_reader:
                # Append chunk to output file
                chunk.to_csv(output_file, mode='a', header=False, index=False)
                total_rows += len(chunk)
                
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue
    
    print(f"\nSuccessfully combined {len(txt_files)} files.")
    print(f"Total rows: {total_rows}")
    print(f"Output file: {output_file}")

# Usage
folder_path = 'taxi_data_full'  # Replace with your actual path
combine_txt_files_to_csv(folder_path)
import os
import pandas as pd
from glob import glob

def extract_well_metrics(input_path):
    # Find all well_metrics_all_wells.csv files in subdirectories
    pattern = os.path.join(input_path, "**/well_metrics_all_wells.csv")
    all_files = glob(pattern, recursive=True)
    
    all_dfs = []
    for file_path in all_files:
        # Read the CSV
        df = pd.read_csv(file_path, header=None)
        
        # Get the parent folder name (the experiment folder)
        folder_name = os.path.basename(os.path.dirname(file_path))
        
        # Create dataframe with desired columns
        result_df = pd.DataFrame({
            'block': df[0],         # First column (block)
            'well': df[1],          # Second column (well name)
            folder_name: df[3]      # Fourth column (metric)
        })
        
        all_dfs.append(result_df)
    
    # Combine all dataframes
    final_df = pd.concat(all_dfs, axis=1)
    
    # Remove duplicate block and well columns
    cols_to_keep = ['block', 'well'] + [col for col in final_df.columns if col not in ['block', 'well']]
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]
    final_df = final_df[cols_to_keep]
    
    # Save to CSV
    output_path = os.path.join(input_path, "extracted_metrics_by_wells.csv")
    print(f"Saving to {output_path}")
    final_df.to_csv(output_path, index=False, encoding='utf-8')
    
    return final_df

# Example usage:
input_path = "analysis_output_1126_final"
result = extract_well_metrics(input_path)

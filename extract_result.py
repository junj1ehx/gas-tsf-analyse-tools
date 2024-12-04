import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import re

def load_data(folder_path):
    """Load prediction and true values from numpy files."""
    metrics = np.load(os.path.join(folder_path, 'metrics.npy'), allow_pickle=True)[:3]
    pred = np.load(os.path.join(folder_path, 'pred.npy'))
    true = np.load(os.path.join(folder_path, 'true.npy'))
    return metrics, pred, true

def extract_results(results_folder, output_folder):
    """Extract raw results from all models and save to CSV files."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create a CSV file to store model information
    model_info = []
    
    for subdir, _, _ in tqdm(os.walk(results_folder)):
        if 'metrics.npy' in os.listdir(subdir):
            model_name = os.path.basename(subdir)
            print(f"\nProcessing {model_name}")
            
            # Load data
            metrics, pred, true = load_data(subdir)
            true = true[:, 0, -1]
            pred = pred[:, 0, -1]
            
            # Extract model parameters
            sl = int(re.search(r'sl(\d+)', model_name).group(1))
            ll = int(re.search(r'll(\d+)', model_name).group(1))
            pl = int(re.search(r'pl(\d+)', model_name).group(1))
            
            # Create model-specific folder
            model_output_folder = os.path.join(output_folder, model_name)
            os.makedirs(model_output_folder, exist_ok=True)
            
            # Save predictions and true values
            results_df = pd.DataFrame({
                'true_values': true.flatten(),
                'predicted_values': pred.flatten()
            })
            results_df.to_csv(os.path.join(model_output_folder, 'raw_results.csv'), index=True)
            
            # Store model information
            model_info.append({
                'model_name': model_name,
                'sequence_length': sl,
                'label_length': ll,
                'prediction_length': pl,
                'mae': metrics[0],
                'mse': metrics[1],
                'rmse': metrics[2]
            })
            
            # If well information exists, save well-specific results
            for block in ["桃49区块", "苏14区块", "苏59区块"]:
                if block in model_name:
                    well_info_path = f'outputs/dataset/blocks/{block}_well_info_seq{sl}_label{ll}_pred{pl}.csv'
                    if os.path.exists(well_info_path):
                        well_df = pd.read_csv(well_info_path)
                        well_df = well_df[well_df['set'] == 'test']
                        
                        for _, well_info in well_df.iterrows():
                            well_name = well_info['well']
                            start_idx = well_info['start_point']
                            end_idx = well_info['end_point']
                            
                            well_results = pd.DataFrame({
                                'true_values': true[start_idx:end_idx],
                                'predicted_values': pred[start_idx:end_idx]
                            })
                            well_results.to_csv(
                                os.path.join(model_output_folder, f'{well_name}_results.csv'),
                                index=True
                            )
    
    # Save model information
    model_info_df = pd.DataFrame(model_info)
    model_info_df.to_csv(os.path.join(output_folder, 'model_summary.csv'), index=False)

if __name__ == "__main__":
    results_folder = '/home/gbu-hkx/project/gas/Time-Series-Library-gas/results/'
    output_folder = 'extracted_results'
    extract_results(results_folder, output_folder)

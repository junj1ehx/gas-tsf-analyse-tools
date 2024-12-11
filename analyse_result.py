import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import re
import seaborn as sns

def load_data(folder_path):
    metrics = np.load(os.path.join(folder_path, 'metrics.npy'), allow_pickle=True)[:3] # mae, mse, rmse
    pred = np.load(os.path.join(folder_path, 'pred.npy'))
    true = np.load(os.path.join(folder_path, 'true.npy'))
    true_point = true[:, 0, -1]
    pred_point = pred[:, 0, -1]
    # true_point = true[:, 1, -1]
    # pred_point = pred[:, 1, -1]
    return metrics, pred_point, true_point

def plot_predictions(pred=[], true=[], length=-1, output_path=''):
    plt.figure(figsize=(12, 6))
    # Convert to numpy arrays if they aren't already
    pred = np.array(pred)
    true = np.array(true)
    
    # Handle NaN and infinite values
    pred = np.nan_to_num(pred, nan=0.0, posinf=None, neginf=None)
    true = np.nan_to_num(true, nan=0.0, posinf=None, neginf=None)
    
    # Create x-axis values
    x = range(len(true[:length]))
    
    plt.plot(x, true[:length], label='True', alpha=0.7, linewidth=2)
    plt.plot(x, pred[:length], label='Predicted', alpha=0.7, linewidth=2)
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title('Predicted vs True Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add some padding to the y-axis
    plt.margins(y=0.1)
    
    # Ensure tight layout
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()




def calculate_additional_metrics(pred, true):
    # alignment
    pred = pred[1:]
    true = true[0:-1]

    """Calculate metrics with NaN handling and validation"""
    # Print initial diagnostics
    print(f"\nDiagnostics before metric calculation:")
    print(f"Pred shape: {pred.shape}, True shape: {true.shape}")
    print(f"NaN in pred: {np.isnan(pred).sum()}, NaN in true: {np.isnan(true).sum()}")
    
    # Handle NaN values by removing corresponding pairs
    valid_mask = ~(np.isnan(pred) | np.isnan(true))
    pred_clean = pred[valid_mask]
    true_clean = true[valid_mask]
    
    print(f"Valid samples after NaN removal: {valid_mask.sum()} out of {len(pred)}")
    
    if len(pred_clean) == 0:
        print("WARNING: No valid samples after NaN removal!")
        return {
            'MSE': np.nan,
            'MAE': np.nan,
            'R2': np.nan,
            'Correlation': np.nan,
            'MAPE': np.nan,
            'RMSE': np.nan
        }
    
    try:
        mse = mean_squared_error(true_clean, pred_clean)
        mae = mean_absolute_error(true_clean, pred_clean)
        r2 = r2_score(true_clean, pred_clean)
        correlation = np.corrcoef(true_clean, pred_clean)[0, 1]
        
        # Handle division by zero in MAPE calculation
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((true_clean - pred_clean) / true_clean)) * 100
            mape = np.nan_to_num(mape, nan=np.inf)  # Replace NaN with inf for zero true values
        
        rmse = np.sqrt(mse)
        
        return {
            'MSE': mse,
            'MAE': mae,
            'R2': r2,
            'Correlation': correlation,
            'MAPE': mape,
            'RMSE': rmse
        }
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        print(f"Pred range: [{np.min(pred_clean)}, {np.max(pred_clean)}]")
        print(f"True range: [{np.min(true_clean)}, {np.max(true_clean)}]")
        raise

def plot_error_distribution(pred, true, output_path):
    errors = true - pred
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.7)
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.savefig(output_path)
    plt.close()

def plot_residuals(pred, true, output_path):
    residuals = true - pred
    plt.figure(figsize=(10, 6))
    plt.scatter(pred, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.savefig(output_path)
    plt.close()

def plot_cumulative_error(pred, true, output_path):
    cumulative_error = np.cumsum(true - pred)
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_error, alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('Cumulative Error')
    plt.title('Cumulative Error Plot')
    plt.savefig(output_path)
    plt.close()

def plot_scatter(pred, true, output_path):
    plt.figure(figsize=(10, 6))
    plt.scatter(true, pred, alpha=0.5)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Scatter Plot of Predictions vs True Values')
    plt.savefig(output_path)
    plt.close()

def plot_boxplot_errors(pred, true, output_path):
    errors = true - pred
    plt.figure(figsize=(10, 6))
    plt.boxplot(errors, vert=False)
    plt.xlabel('Error')
    plt.title('Box Plot of Errors')
    plt.savefig(output_path)
    plt.close()

def save_metrics_to_csv(metrics, output_csv):
    df = pd.DataFrame(metrics)
    df.to_csv(output_csv, index=False)

# def plot_predictions_for_well(block_key, pred, true, well_df, output_folder, seq_len, label_len, pred_len):
#     start_point_tmp = 0
#     end_point_tmp = 0
#     # open an csv file to record the block name, well name, and the calculated metrics for each well, the file can be added not overwrite
#     with open(os.path.join(output_folder, 'well_metrics_all_wells.csv'), 'a') as f:
#         for _, well_info in well_df.iterrows():
#             """Plot predictions for a specific well sequence."""
#             end_point_tmp = start_point_tmp + well_info['end_point'] - well_info['start_point']
#             well_name = well_info['well']
            
#             # Create well sequence plot
#             plt.figure(figsize=(12, 6))
#             # print(well_info)
#             # print(true.shape, pred.shape)
#             # print(start_point, end_point)
#             # # print all value and read
#             # print(true)
#             # print(pred)
#             x = range(well_info['end_point'] - well_info['start_point'])
#             print(well_info)
#             # print(start_point_tmp, end_point_tmp)
#             plt.plot(x, true[start_point_tmp:end_point_tmp], label='True', alpha=0.7, linewidth=2)
#             plt.plot(x, pred[start_point_tmp:end_point_tmp], label='Predicted', alpha=0.7, linewidth=2)
            
#             plt.xlabel('Months')
#             plt.ylabel('Value')
#             plt.title(f'{well_name}')
#             plt.legend()
#             plt.grid(True, alpha=0.3)
#             plt.margins(y=0.1)
#             plt.tight_layout()
            
#             # Save plot
#             output_path = os.path.join(output_folder, f"{well_name}_plot.png")
#             plt.savefig(output_path, dpi=300, bbox_inches='tight')
#             plt.close()
#             # output true[start_point:end_point] to csv, start_point:end_point as index
#             true_df = pd.DataFrame(true[start_point_tmp:end_point_tmp], index=range(start_point_tmp, end_point_tmp))
#             true_df.to_csv(os.path.join(output_folder, f"{well_name}_true.csv"), index=True)
#             pred_df = pd.DataFrame(pred[start_point_tmp:end_point_tmp], index=range(start_point_tmp, end_point_tmp))
#             pred_df.to_csv(os.path.join(output_folder, f"{well_name}_pred.csv"), index=True)
            
#             # write the metrics to the csv file
#             metrics = calculate_additional_metrics(pred[start_point_tmp:end_point_tmp], true[start_point_tmp:end_point_tmp])
#             f.write(f"{block_key},{well_name},{metrics['MAE']},{metrics['MSE']},{metrics['RMSE']}\n")


#             start_point_tmp = end_point_tmp


def plot_predictions_for_well(block_key, pred, true, well_df, output_folder, seq_len, label_len, pred_len):
    with open(os.path.join(output_folder, 'well_metrics_all_wells.csv'), 'a') as f:
        matrix = {}
        for i in range(len(well_df)):
            # pred and true value for each well
            if well_df['well'].iloc[i] not in matrix.keys():
                matrix[well_df['well'].iloc[i]] = []
            matrix[well_df['well'].iloc[i]].append([pred[i], true[i]])
        
        # plot the pred and true value for each well
        for well_name, values in matrix.items():
            # Create well sequence plot
            plt.figure(figsize=(12, 6))
            
            # Extract true and predicted values for the current well
            true_values = np.array([v[1] for v in values])
            pred_values = np.array([v[0] for v in values])
            
            # Define the x-axis range
            x = range(len(true_values))
            
            # Plot true and predicted values
            plt.plot(x, true_values, label='True', alpha=0.7, linewidth=2)
            plt.plot(x, pred_values, label='Predicted', alpha=0.7, linewidth=2)
            
            plt.xlabel('Months')
            plt.ylabel('Value')
            plt.title(f'{well_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.margins(y=0.1)
            plt.tight_layout()
            
            # Save plot
            output_path = os.path.join(output_folder, f"{well_name}_plot.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Output true and predicted values to CSV
            true_df = pd.DataFrame(true_values, index=range(len(true_values)))
            true_df.to_csv(os.path.join(output_folder, f"{well_name}_true.csv"), index=True)
            pred_df = pd.DataFrame(pred_values, index=range(len(pred_values)))
            pred_df.to_csv(os.path.join(output_folder, f"{well_name}_pred.csv"), index=True)
            
            # Write the metrics to the CSV file
            metrics = calculate_additional_metrics(pred_values, true_values)
            f.write(f"{block_key},{well_name},{metrics['MAE']},{metrics['MSE']},{metrics['RMSE']}\n")
            
def analyze_results(results_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    all_metrics = []
    for subdir, _, _ in tqdm(os.walk(results_folder)):
        if 'metrics.npy' in os.listdir(subdir):
            metrics, pred, true = load_data(subdir)
            # print(pred[:200])
            # # print true with 6 decimal places
            # print(true[:200].round(6))

            
            # Create block-specific output folder
            block_name = os.path.basename(subdir)
            block_output_folder = os.path.join(output_folder, block_name)
            os.makedirs(block_output_folder, exist_ok=True)
            
            # Add data validation and print statistics
            if np.any(np.isnan(pred)) or np.any(np.isnan(true)):
                print(f"Warning: NaN values found in {subdir}")
            if np.any(np.isinf(pred)) or np.any(np.isinf(true)):
                print(f"Warning: Infinite values found in {subdir}")
            
            print(f"\nAnalyzing {block_name}:")
            # get sl, ll, and pl from block_name
            sl = int(re.search(r'sl(\d+)', block_name).group(1))
            ll = int(re.search(r'll(\d+)', block_name).group(1))
            pl = int(re.search(r'pl(\d+)', block_name).group(1))

            # Load well information
            # well_info_files = {
            #     '桃49区块': f'outputs/dataset/blocks/桃49区块_well_info_seq{sl}_label{ll}_pred{pl}.csv',
            #     '苏14区块': f'outputs/dataset/blocks/苏14区块_well_info_seq{sl}_label{ll}_pred{pl}.csv',
            #     '苏59区块': f'outputs/dataset/blocks/苏59区块_well_info_seq{sl}_label{ll}_pred{pl}.csv'
            # }
            well_location_files = {
                '桃49区块': f'outputs/dataset/blocks/桃49区块_data_location_seq{sl}_label{ll}_pred{pl}.csv',
                '苏14区块': f'outputs/dataset/blocks/苏14区块_data_location_seq{sl}_label{ll}_pred{pl}.csv',
                '苏59区块': f'outputs/dataset/blocks/苏59区块_data_location_seq{sl}_label{ll}_pred{pl}.csv'
            }
            print(f"sl: {sl}, ll: {ll}, pl: {pl}")
            print(f"Prediction range: [{np.min(pred):.2f}, {np.max(pred):.2f}]")
            print(f"True range: [{np.min(true):.2f}, {np.max(true):.2f}]")
            
            # Calculate metrics
            additional_metrics = calculate_additional_metrics(pred, true)
            metrics_dict = {'MAE_ori': metrics[0], 'MSE_ori': metrics[1], 'RMSE_ori': metrics[2]}
            data_dict = {'model': block_name}
            all_metrics.append({**data_dict, **metrics_dict, **additional_metrics})

            # # Plot overall predictions
            # plot_predictions(pred=pred, true=true, length=-1, 
            #                output_path=os.path.join(block_output_folder, f"{block_name}_overall.png"))
            
            # # Plot predictions
            # plot_output_path = os.path.join(output_folder, f"{os.path.basename(subdir)}_plot.png")
            # plot_output_path_1000 = os.path.join(output_folder, f"{os.path.basename(subdir)}_plot_1000.png")
            # plot_predictions(pred=pred, true=true, length=1000, output_path=plot_output_path_1000)
            # plot_predictions(pred=pred, true=true, length=-1, output_path=plot_output_path)

            # # Plot additional analyses
            # # plot_error_distribution(pred, true, os.path.join(output_folder, f"{os.path.basename(subdir)}_error_dist.png"))
            # # plot_residuals(pred, true, os.path.join(output_folder, f"{os.path.basename(subdir)}_residuals.png"))
            # # plot_cumulative_error(pred, true, os.path.join(output_folder, f"{os.path.basename(subdir)}_cumulative_error.png"))
            # # plot_scatter(pred, true, os.path.join(output_folder, f"{os.path.basename(subdir)}_scatter.png"))
            # # plot_boxplot_errors(pred, true, os.path.join(output_folder, f"{os.path.basename(subdir)}_boxplot_errors.png"))
            
            # for block_key, well_location_path in well_location_files.items():
            #     if block_key in block_name:
            #         # read the csv file and give the column names with 'well', 'set', 'location' and reset the index
            #         well_df = pd.read_csv(well_location_path, names=['well', 'set', 'location']).reset_index(drop=True)
            #         well_df = well_df[well_df['set'] == 2]
                    
            #         plot_predictions_for_well(block_key, pred, true, well_df, block_output_folder, sl, ll, pl)

    # Save metrics to CSV
    metrics_csv_path = os.path.join(output_folder, 'all_metrics.csv')
    save_metrics_to_csv(all_metrics, metrics_csv_path)

def calculate_well_metrics(pred, true):

    """Calculate MSE and MAE for a single well's predictions"""
    # Handle NaN values
    valid_mask = ~(np.isnan(pred) | np.isnan(true))
    pred_clean = pred[valid_mask]
    true_clean = true[valid_mask]
    
    if len(pred_clean) == 0:
        return np.nan, np.nan
        
    try:
        # alignment
        pred_clean = pred_clean[1:]
        true_clean = true_clean[0:-1]
        mse = mean_squared_error(true_clean, pred_clean)
        mae = mean_absolute_error(true_clean, pred_clean)
        return mse, mae
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return np.nan, np.nan

def compare_well_performances(results_folder, output_folder):
    """Compare algorithm performances for each well across different models."""
    well_metrics = {}
    well_performance_metrics = {}
    
    # First, collect all predictions and true values for each well from different models
    for subdir, _, _ in os.walk(results_folder):
        if 'metrics.npy' not in os.listdir(subdir):
            continue
            
        model_name = os.path.basename(subdir)
        _, pred, true = load_data(subdir)
        
        # Extract sequence length parameters from model name
        sl = int(re.search(r'sl(\d+)', model_name).group(1))
        ll = int(re.search(r'll(\d+)', model_name).group(1))
        pl = int(re.search(r'pl(\d+)', model_name).group(1))

        # Determine which block this model is for
        block_name = None
        for block in ["苏59区块", "苏14区块", "苏49区块"]:
            if block in model_name:
                block_name = block
                break
        
        if not block_name:
            continue
            
        # # Load well information
        # well_info_path = f'outputs/dataset/blocks/{block_name}_well_info_seq{sl}_label{ll}_pred{pl}.csv'
        # well_df = pd.read_csv(well_info_path)
        # well_df = well_df[well_df['set'] == 'test']  # Only use test set data
        
        well_location_files = {
                '桃49区块': f'outputs/dataset/blocks/桃49区块_data_location_seq{sl}_label{ll}_pred{pl}.csv',
                '苏14区块': f'outputs/dataset/blocks/苏14区块_data_location_seq{sl}_label{ll}_pred{pl}.csv',
                '苏59区块': f'outputs/dataset/blocks/苏59区块_data_location_seq{sl}_label{ll}_pred{pl}.csv'
            }        

        for block_key, well_location_path in well_location_files.items():
            if block_key in block_name:
                # read the csv file and give the column names with 'well', 'set', 'location' and reset the index
                well_df = pd.read_csv(well_location_path, names=['well', 'set', 'location']).reset_index(drop=True)
                well_df = well_df[well_df['set'] == 2]


        

        matrix = {}
        for i in range(len(well_df)):
            # pred and true value for each well
            if well_df['well'].iloc[i] not in matrix.keys():
                matrix[well_df['well'].iloc[i]] = []
            matrix[well_df['well'].iloc[i]].append([pred[i], true[i]])
        
        # plot the pred and true value for each well
        for well_name, values in matrix.items():
            # Initialize dictionaries for new wells
            if well_name not in well_metrics:
                well_metrics[well_name] = {}  # Initialize dict for new well
            if well_name not in well_performance_metrics:
                well_performance_metrics[well_name] = {}
            
            # Extract true and predicted values for the current well
            true_values = np.array([v[1] for v in values])
            pred_values = np.array([v[0] for v in values])

            mse, mae = calculate_well_metrics(pred_values, true_values)
            
            # Store the metrics
            well_performance_metrics[well_name][model_name] = {
                'MSE': mse,
                'MAE': mae
            }
            
            well_metrics[well_name][model_name] = {
                'pred': pred_values,
                'true': true_values,
                'sl': sl,
                'block': block_name
            }
            

    # Initialize summary data ONCE, before processing wells
    summary_data = []
    # Create comparison plots for each well use tqdm
    for well_name, models in tqdm(well_metrics.items(), desc="Processing wells"):
        print(well_name)
        plt.figure(figsize=(12, 6))
        plt.style.use('seaborn')
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
        # select sharp color on having different model names
        model_names = ['PatchTST', 'SVM', 'ANN', 'RNN', 'LSTM', 'TimeMixer', 'TimesNet', 'iTransformer']

        model_colors = {name: color for name, color in zip(model_names, plt.cm.tab20(np.linspace(0, 1, len(model_names))))}
        # highlight TimeMixer and LSTM
        model_colors['TimeMixer'] = 'Blue'
        model_colors['LSTM'] = 'Green'

        # Get model data that true is the longest
        true_lengths = [len(data['true']) for data in models.values()]
        model_data = list(models.values())[np.argmax(true_lengths)]
        # get the pl of selected model data
        model_sl = model_data['sl']
        # Set font for Chinese characters
        plt.rcParams['font.family'] = ['SimHei']  # For Chinese characters
        plt.rcParams['axes.unicode_minus'] = False    # For minus sign

        # Read historical data from original CSV
        historical_csv_path = f'outputs/dataset/{well_name}.csv'
        historical_df = pd.read_csv(historical_csv_path)
        historical_values = historical_df['OT'].values[:sl]  # Replace 'target_column' with your actual column name

        # Combine historical and true values
        combined_values = np.concatenate([historical_values, model_data['true']])

        # Plot true values first
        plt.plot(range(len(combined_values)), combined_values, 
                label='True Values', 
                color='red',
                linestyle='-',
                linewidth=1,
                zorder=10)


        # Plot predictions for each model
        for (model_path, data), color in zip(models.items(), colors):
            # Extract model name and ft type
            parts = model_path.split('_')
            
            # More flexible model name extraction
            try:
                model_name = next(part for part in parts if part in ['PatchTST', 'SVM', 'ANN', 'RNN', 'LSTM', 'TimeMixer', 'TimesNet', 'iTransformer'])
            except StopIteration:
                print(f"Warning: Could not find expected model name in path: {model_path}")
                model_name = parts[0] if parts else 'Unknown'
            
            # More flexible ft_type extraction
            try:
                ft_type = next(part for part in parts if part.startswith('ft'))[2:]
            except StopIteration:
                ft_type = 'default'
            
            # Extract parameters
            sl = int(re.search(r'sl(\d+)', model_path).group(1))
            ll = int(re.search(r'll(\d+)', model_path).group(1))
            pl = int(re.search(r'pl(\d+)', model_path).group(1))
            # Create label
            label = f'{model_name}-ft{ft_type} (sl{sl}_ll{ll}_pl{pl})'
            # align the data to the true values in model data by checking if true values are equal to the data
            # assert len(data['true']) == len(data['pred'])
            adjusted_pred = data['pred']
            x = range(sl - model_sl + sl, len(adjusted_pred)-1 + sl)
            # adjusted_pred = adjusted_pred[:model_pl]
            if len(adjusted_pred) > 0:  # Only plot if we have data
                # use model_colors to get the color
                color = model_colors[model_name]
                plt.plot(x, adjusted_pred[1:len(x)+1], 
                        label=label, 
                        color=color, 
                        alpha=0.7, 
                        linewidth=1,
                        zorder=5)
        
        plt.xlabel('Months', fontsize=10)
        plt.ylabel('Daily Production', fontsize=10)
        plt.title(f'{well_name}', fontsize=12, pad=15)
        
        # Improve grid and background
        plt.grid(True, alpha=0.3)
        plt.gca().set_facecolor('#f8f9fa')
        
        # Improve legend - put true values first
        handles, labels = plt.gca().get_legend_handles_labels()
        true_idx = labels.index('True Values')
        handles = [handles[true_idx]] + handles[:true_idx] + handles[true_idx+1:]
        labels = [labels[true_idx]] + labels[:true_idx] + labels[true_idx+1:]
        
        plt.legend(handles, labels,
                  bbox_to_anchor=(1.05, 1), 
                  loc='upper left', 
                  fontsize=9, 
                  frameon=True, 
                  facecolor='white', 
                  edgecolor='none', 
                  shadow=True)
        
        plt.margins(x=0.02)
        plt.tight_layout()
        
        # make sure the output folder exists
        os.makedirs(os.path.join(output_folder, 'well_comparisons'), exist_ok=True)
        # Save with high quality
        plt.savefig(
            os.path.join(output_folder, 'well_comparisons', f'{model_data["block"]}_{well_name}_comparison.png'),
            bbox_inches='tight',
            dpi=300,
            facecolor='white',
            edgecolor='none'
        )
        plt.close()
        
        # Collect metrics for this well
        for model_name, data in models.items():
            # Add check for data length before calculating metrics
            if len(data['pred']) > 0 and len(data['true']) > 0:
                # Calculate metrics for this model's predictions
                metrics = calculate_additional_metrics(data['pred'], data['true'])
                metrics_row = {
                    'Well': well_name,
                    'Model': model_name,
                    **metrics
                }
                summary_data.append(metrics_row)
    
    # Save summary metrics to CSV
    summary_df = pd.DataFrame(summary_data)
    if not os.path.exists(os.path.join(output_folder, 'well_comparisons')):
        os.makedirs(os.path.join(output_folder, 'well_comparisons'))
    summary_df.to_csv(os.path.join(output_folder, 'well_comparisons', 'well_performance_summary.csv'),
                     index=False)
    
    # Create pivot tables for easy comparison
    for metric in ['MSE', 'MAE', 'R2', 'RMSE', 'MAPE']:
        pivot_df = summary_df.pivot(index='Well', columns='Model', values=metric)
        pivot_df.to_csv(os.path.join(output_folder, 'well_comparisons', f'well_comparison_{metric}.csv'))

def well_statistics(results_folder, output_folder):
    """Calculate and output statistics for each well."""
    well_stats = {}
    
    # First, collect all true values for each well
    for subdir, _, _ in os.walk(results_folder):
        if 'metrics.npy' not in os.listdir(subdir):
            continue
            
        model_name = os.path.basename(subdir)
        _, _, true = load_data(subdir)  # We only need true values for statistics
        
        # Extract sequence length parameters
        sl = int(re.search(r'sl(\d+)', model_name).group(1))
        ll = int(re.search(r'll(\d+)', model_name).group(1))
        pl = int(re.search(r'pl(\d+)', model_name).group(1))
        
        # Determine which block this model is for
        block_name = None
        for block in ["苏14区块", "苏59区块", "苏49区块"]:
            if block in model_name:
                block_name = block
                break
        
        if not block_name:
            continue
            
        # Load well information
        well_info_path = f'outputs/dataset/blocks/{block_name}_well_info_seq{sl}_label{ll}_pred{pl}.csv'
        well_df = pd.read_csv(well_info_path)
        well_df = well_df[well_df['set'] == 'test']  # Only use test set data
        
        # Process each well
        for _, well_info in well_df.iterrows():
            well_name = well_info['well']
            length = well_info['end_point'] - well_info['start_point'] + 1
        
            
            if well_name not in well_stats:
                well_stats[well_name] = {
                    'values': well_name,
                    'length': length,
                    'block': block_name
                }
            

    # Create DataFrame from statistics
    stats_data = []
    for well_name, stats in well_stats.items():
        stats_data.append({
            'Well': well_name,
            'Block': stats['block'],
            'Length': stats['length']
        })
    
    # Convert to DataFrame and save
    stats_df = pd.DataFrame(stats_data)
    
    # Sort by block and well name
    stats_df = stats_df.sort_values(['Block', 'Well'])
    
    # Save to CSV
    os.makedirs(os.path.join(output_folder, 'well_statistics'), exist_ok=True)
    stats_df.to_csv(os.path.join(output_folder, 'well_statistics', 'well_statistics.csv'), 
                    index=False)
    
    # Print summary statistics
    print("\nWell Statistics Summary:")
    print(f"Total number of wells: {len(stats_df)}")
    print("\nBy Block:")
    print(stats_df.groupby('Block').size())
    print("\nOverall Statistics:")
    print(stats_df.describe())

if __name__ == "__main__":
    plt.rcParams['font.family'] = 'SimHei'    
    results_folder = '/home/gbu-hkx/project/gas/Time-Series-Library-gas/1211_result_32/results_for_pic/6_3'
    # results_folder = "/home/gbu-hkx/project/gas/Time-Series-Library/results_backup/results/long_term_forecast_/home/gbu-hkx/project/gas/Time-Series-Library/dataset/gas_all/"
    output_folder = 'analysis_outputs_1211/6_3_new'
    analyze_results(results_folder, output_folder)
    compare_well_performances(results_folder, output_folder)
    # well_statistics(results_folder, output_folder)
    

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from pathlib import Path

# Set up plotting style
plt.style.use('seaborn')
sns.set_palette("husl")
plt.rcParams['font.size'] = 8
# Path to data files
data_path = '/home/gbu-hkx/project/gas/data/outputs/dataset/blocks'


import matplotlib.font_manager
print([f.name for f in matplotlib.font_manager.fontManager.ttflist])
plt.rcParams['font.family'] = 'SimHei'
# Create output directories for plots
plot_dir = Path('plots')
plot_dir.mkdir(exist_ok=True)
stats_dir = Path('statistics') 
stats_dir.mkdir(exist_ok=True)

def analyze_block_data(file_path):
    # Read data
    df = pd.read_csv(file_path)
    block_name = os.path.splitext(os.path.basename(file_path))[0]
    
    print(f"\nAnalyzing {block_name}")
    print("-" * 50)
    
    # Basic statistics
    stats = df.describe()
    stats.to_csv(stats_dir / f"{block_name}_statistics.csv")
    
    # Time series analysis
    df['date'] = pd.to_datetime(df['date'])
    
    # Plot 1: Production trends
    plt.figure(figsize=(15, 8))
    plt.plot(df['date'], df['OT'], label='Daily Gas Production')
    plt.title(f'{block_name} - Daily Gas Production Over Time')
    plt.xlabel('Date')
    plt.ylabel('Daily Gas Production (10^4m³)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / f"{block_name}_production_trend.png")
    plt.close()
    
    # Plot 2: Pressure relationships
    plt.figure(figsize=(12, 8))
    plt.scatter(df['油压(MPa)'], df['OT'], alpha=0.5, label='Oil Pressure vs Production')
    plt.scatter(df['套压(MPa)'], df['OT'], alpha=0.5, label='Casing Pressure vs Production')
    plt.xlabel('Pressure (MPa)')
    plt.ylabel('Daily Gas Production (10^4m³)')
    plt.title(f'{block_name} - Pressure vs Production Relationship')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / f"{block_name}_pressure_production.png")
    plt.close()
    
    # Plot 3: Monthly production boxplot
    df['month'] = df['date'].dt.month
    plt.figure(figsize=(15, 8))
    sns.boxplot(x='month', y='OT', data=df)
    plt.title(f'{block_name} - Monthly Production Distribution')
    plt.xlabel('Month')
    plt.ylabel('Daily Gas Production (10^4m³)')
    plt.tight_layout()
    plt.savefig(plot_dir / f"{block_name}_monthly_distribution.png")
    plt.close()
    
    # Calculate correlations
    corr = df.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title(f'{block_name} - Correlation Matrix')
    plt.tight_layout()
    plt.savefig(plot_dir / f"{block_name}_correlation_matrix.png")
    plt.close()
    
    # Calculate and print key metrics
    total_production = df['OT'].sum()
    avg_production = df['OT'].mean()
    max_production = df['OT'].max()
    variance = df['OT'].var()
    # calculate the data points where OT is 0
    zero_production_days = df[df['OT'] == 0].shape[0]
    # average production without zero production days
    avg_production_without_zero = df[df['OT'] != 0]['OT'].mean()
    # calculate the number of wells
    num_wells = df['井号'].nunique()
    metrics = {
        'Total Production (10^4m³)': total_production,
        'Average Daily Production (10^4m³)': avg_production,
        'Average Daily Production without Zero Production Days (10^4m³)': avg_production_without_zero,
        'Maximum Daily Production (10^4m³)': max_production,
        'Variance of Daily Production (10^4m³)': variance,
        'Average Oil Pressure (MPa)': df['油压(MPa)'].mean(),
        'Average Casing Pressure (MPa)': df['套压(MPa)'].mean(),
        'Production Days': df['生产天数(d)'].sum(),
        'Data Points': len(df),
        'Zero Production Days': zero_production_days,
        'Number of Wells': num_wells
    }
    
    pd.Series(metrics).to_csv(stats_dir / f"{block_name}_key_metrics.csv")
    
    return metrics

# Process all files
all_metrics = {}
for file in os.listdir(data_path):
    if file.endswith('.csv') and 'well_info' not in file:
        file_path = os.path.join(data_path, file)
        all_metrics[file] = analyze_block_data(file_path)

# Create comparison plots across all blocks
metrics_df = pd.DataFrame(all_metrics).T

plt.figure(figsize=(15, 8))
metrics_df['Average Daily Production (10^4m³)'].plot(kind='bar')
plt.title('Average Daily Production Comparison Across Blocks')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(plot_dir / "blocks_production_comparison.png")
plt.close()

# Analyze OT (Output Target) characteristics across blocks
ot_stats = {}
for file in os.listdir(data_path):
    if file.endswith('.csv') and 'well_info' not in file:
        file_path = os.path.join(data_path, file)
        df = pd.read_csv(file_path)
        
        # Calculate basic statistics for OT
        stats = {
            'Mean': df['OT'].mean(),
            'Std': df['OT'].std(),
            'Min': df['OT'].min(),
            'Max': df['OT'].max(),
            'Skewness': df['OT'].skew(),
            'Kurtosis': df['OT'].kurtosis()
        }
        ot_stats[file] = stats

# Create DataFrame with OT statistics
ot_stats_df = pd.DataFrame(ot_stats).T

# Plot OT distribution characteristics
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
ot_stats_df['Mean'].plot(kind='bar')
plt.title('Mean OT by Block')
plt.xticks(rotation=45)

plt.subplot(2, 2, 2)
ot_stats_df['Std'].plot(kind='bar')
plt.title('OT Standard Deviation by Block')
plt.xticks(rotation=45)

plt.subplot(2, 2, 3)
ot_stats_df['Skewness'].plot(kind='bar')
plt.title('OT Skewness by Block')
plt.xticks(rotation=45)

plt.subplot(2, 2, 4)
ot_stats_df['Kurtosis'].plot(kind='bar')
plt.title('OT Kurtosis by Block')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(plot_dir / "ot_characteristics.png")
plt.close()

# Save OT statistics
ot_stats_df.to_csv(stats_dir / "ot_statistics.csv")


# Save overall comparison metrics
metrics_df.to_csv(stats_dir / "all_blocks_comparison.csv")

print("\nAnalysis complete. Results saved in 'plots' and 'statistics' directories.")

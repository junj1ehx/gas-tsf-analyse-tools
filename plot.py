import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

def read_and_transform_gas_data(folder_path='raw_data', output_folder='outputs'):
    # Initialize an empty list to store dataframes
    dataframes = []

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Create output folder if it doesn't exist
    output_figures_folder = os.path.join(output_folder, 'figures')
    if not os.path.exists(output_figures_folder):
        os.makedirs(output_figures_folder)
    

    # Iterate through all files in the gas folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.xls'):
            # Construct the full file path
            file_path = os.path.join(folder_path, file_name)
            # Read the Excel file into a DataFrame
            df = pd.read_excel(file_path, engine='xlrd', skiprows=[0])
            dataframes.append(df)

    # Concatenate all the dataframes into one
    all_data = pd.concat(dataframes, ignore_index=True)

    # Convert '年月' (formatted as yyyymm) to a datetime format
    all_data['年月'] = all_data['年月'].apply(lambda x: datetime.strptime(str(x), '%Y%m'))

    # Group by '区块' (Block)
    block_groups = all_data.groupby('区块')

    for block, block_data in block_groups:
        # Further group by '井号' (Well Number) within each block
        well_groups = block_data.groupby('井号')
        
        # plt.figure(figsize=(10, 6))
        # Rotate the x-axis labels for better readability
        plt.rcParams['font.sans-serif']=['Arial']
        plt.xlabel('Year-Month', fontsize=12)
        plt.ylabel('Daily Gas Production (10^4 m³)', fontsize=12)
        plt.xticks(rotation=45)
        plt.title(f'Block: {block}', fontsize=14)
        plt.grid(True)
        
        for well, well_data in well_groups:
            # Plotting the time series for each unique well (井号)
            print(well_data)
            assert 1==0
            plt.plot(well_data['年月'], well_data['日产气(10^4m³)'], linestyle='-', label=well)
            well_output_folder = os.path.join()
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
        
        # Save the plot as a PNG file
        plot_filename = f"{output_figures_folder}/Block_{block}.png"
        plt.legend()
        plt.savefig(plot_filename, bbox_inches='tight')
        plt.close()

        print(f"Plot saved for Block: {block} -> {plot_filename}")
        

# Call the function to execute the process
read_and_transform_gas_data()

    # Example of a simple transformation: rename columns
    # all_data.rename(columns={
    #     '气田': 'GasField',
    #     '作业区': 'OperationArea',
    #     '区块': 'Block',
    #     '站库': 'Station',
    #     '井号': 'WellNumber',
    #     '年度': 'Year',
    #     '年月': 'YearMonth',
    #     '生产天数(d)': 'ProductionDays',
    #     '油压(MPa)': 'OilPressure_MPa',
    #     '套压(MPa)': 'CasingPressure_MPa',
    #     '日产气(10^4m³)': 'DailyGasProduction_104m3',
    #     # Add other column mappings as needed
    # }, inplace=True)

    # Save the transformed data to a CSV file
#     all_data.to_csv(output_file, index=False)

#     print(f"Data successfully processed and saved to {output_file}")

# # Call the function to execute the process
# read_and_transform_gas_data()
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import random
def read_and_transform_gas_data(folder_path='raw_data', output_folder='outputs'):
    # Initialize an empty list to store dataframes
    dataframes = []

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Create output figures folder if it doesn't exist
    output_data_folder = os.path.join(output_folder, 'dataset')
    if not os.path.exists(output_data_folder):
        os.makedirs(output_data_folder)

    # Iterate through all files in the gas folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.xls'):
            # Construct the full file path
            file_path = os.path.join(folder_path, file_name)
            # Read all sheets in the Excel file into a DataFrame
            # Read all sheets from the Excel file
            excel_file = pd.ExcelFile(file_path)
            # Read each sheet and combine into single DataFrame
            df = pd.concat([pd.read_excel(excel_file, sheet_name=sheet, skiprows=[0]) 
                          for sheet in excel_file.sheet_names], ignore_index=True)

            dataframes.append(df)

    # Concatenate all the dataframes into one
    all_data = pd.concat(dataframes, ignore_index=True)
    

    # Convert '年月' (formatted as yyyymm) to a datetime format
    all_data['年月'] = all_data['年月'].apply(lambda x: datetime.strptime(str(x), '%Y%m'))
    
    all_data.rename(columns={'年月': 'date', '日产气(10^4m³)':'OT'},inplace=True)
    print(all_data.head())
    # Group by '区块' (Block)
    block_groups = all_data.groupby('区块')

    for block, block_data in block_groups:
        # Further group by '井号' (Well Number) within each block
        well_groups = block_data.groupby('井号')
        
        for well, well_data in well_groups:
            # Plotting the time series for each unique well (井号)
            well_data_output = well_data[['date', '区块', '井号','生产天数(d)', '油压(MPa)', '套压(MPa)', '日产气水平(10^4m³)', '日产水(m³)','日产油(t)','月产气(10^4m³)','月产水(m³)','月产油(t)','年产气(10^4m³)','年产水(10^4m³)','年产油(10^4t)','累产气(10^8m³)','累产水(10^4m³)','累产油(10^4t)','月工业气量(10^4 m³)','年工业气量(10^4 m³)','累工业气量(10^8 m³)','月商品气量(10^4m³)','年商品气量(10^4m³)','累商品气量(10^8m³)','水气比(m³/10^4m³)','月放空气量(10^4m³)','年放空气量(10^4m³)','累放空气量(10^8m³)','月注醇(m³)','累注醇(m³)', 'OT']]
            output_file = os.path.join(output_data_folder, f'{well}.csv')
            well_data_output.to_csv(output_file, index=False)
            
            # Create blocks folder if it doesn't exist
            blocks_folder = os.path.join(output_data_folder, 'blocks')
            if not os.path.exists(blocks_folder):
                os.makedirs(blocks_folder)
            
            # Get the block name for this well data
            block_name = well_data['区块'].iloc[0]
            
            # Create or append to block-specific dataframe
            if block_name not in locals():
                locals()[block_name] = [well_data_output]
            else:
                locals()[block_name].append(well_data_output)
            
            # After processing all wells, save block data
            if well == list(well_groups.groups.keys())[-1]:  # Convert dict_keys to list
                block_df = pd.concat(locals()[block_name], ignore_index=True)
                block_output_file = os.path.join(blocks_folder, f'{block_name}.csv')
                block_df.to_csv(block_output_file, index=False)
                print(f"Block data saved to {block_output_file}")

            # Collect all well data into a list for concatenation
            if 'all_wells_data' not in locals():
                all_wells_data = [well_data_output]
            else:
                all_wells_data.append(well_data_output)
    # set random seeds
    # random.seed(42)
    # randomly arrange all_wells_data
    # all_wells_data = random.sample(all_wells_data, len(all_wells_data)) 
    # Concatenate all well data and save to CSV
    all_wells_combined = pd.concat(all_wells_data, ignore_index=True)
    all_wells_output = os.path.join(output_data_folder, 'all_wells_combined.csv')
    all_wells_combined.to_csv(all_wells_output, index=False)
    print(f"Combined data from all wells saved to {all_wells_output}")
        
# 气田,作业区,区块,站库,井号,年度,年月,生产天数(d),油压(MPa),套压(MPa),日产气(10^4m³),日产气水平(10^4m³),日产水(m³),日产油(t),月产气(10^4m³),月产水(m³),月产油(t),年产气(10^4m³),年产水(10^4m³),年产油(10^4t),累产气(10^8m³),累产水(10^4m³),累产油(10^4t),月工业气量(10^4 m³),年工业气量(10^4 m³),累工业气量(10^8 m³),月商品气量(10^4m³),年商品气量(10^4m³),累商品气量(10^8m³),水气比(m³/10^4m³),月放空气量(10^4m³),年放空气量(10^4m³),累放空气量(10^8m³),月注醇(m³),累注醇(m³),层位,投资年份,投产日期,井型,井组,采出方式,驱动方式,井口温度(℃),地理位置,备注

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
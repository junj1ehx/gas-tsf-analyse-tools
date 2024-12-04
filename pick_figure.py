import os
import shutil
from pathlib import Path

def pick_figures(source_folders, search_strings, output_base="output_selected_well_figures"):
    """
    Pick image files containing specific strings from source folders and copy to output folders.
    
    Args:
        source_folders (list): List of folder paths to search in
        search_strings (list): List of strings to search for in filenames
        output_base (str): Base output directory name
    """
    # Common image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    
    # Create base output directory if it doesn't exist
    output_dir = Path(output_base)
    output_dir.mkdir(exist_ok=True)
    
    for folder in source_folders:
        block_name = folder.split('/')[-2]
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"Warning: Folder {folder} does not exist")
            continue
            
        # Create output subfolder with source folder name
        folder_output = output_dir / folder_path.name
        folder_output.mkdir(exist_ok=True)
        
        # Search for matching files
        for file in folder_path.rglob("*"):
            if file.suffix.lower() in image_extensions:
                
                # Check if file contains any of the search strings
                if any(s in file.name for s in search_strings):
                    # Create block name folder if not exists
                    block_output = folder_output / block_name
                    block_output.mkdir(exist_ok=True)
                    # Copy file to output folder/block name/file name
                    shutil.copy2(file, block_output / file.name)
                    print(f"Copied: {file.name} to {block_output}")


# Example usage
if __name__ == "__main__":
    # Example folders to search in
    folders_to_search = [
        "analysis_output_1126_for_pic/analysis_output_1126_4_1_all/well_comparisons",
        "analysis_output_1126_for_pic/analysis_output_1126_6_1_all/well_comparisons",
        "analysis_output_1126_for_pic/analysis_output_1126_8_1_all/well_comparisons",
        "analysis_output_1126_for_pic/analysis_output_1126_12_1_all/well_comparisons",
        "analysis_output_1126_for_pic/analysis_output_1126_6_4_1/well_comparisons",
        "analysis_output_1126_for_pic/analysis_output_1126_6_4_2/well_comparisons",
        "analysis_output_1126_for_pic/analysis_output_1126_6_4_3/well_comparisons",
        "analysis_output_1126_for_pic/analysis_output_1126_6_4_4/well_comparisons",
        "analysis_output_1126_for_pic/analysis_output_1126_8_4_1/well_comparisons",
        "analysis_output_1126_for_pic/analysis_output_1126_8_4_2/well_comparisons",
        "analysis_output_1126_for_pic/analysis_output_1126_8_4_3/well_comparisons",
        "analysis_output_1126_for_pic/analysis_output_1126_8_4_4/well_comparisons",
        "analysis_output_1126_for_pic/analysis_output_1126_12_4_1/well_comparisons",
        "analysis_output_1126_for_pic/analysis_output_1126_12_4_2/well_comparisons",
        "analysis_output_1126_for_pic/analysis_output_1126_12_4_3/well_comparisons",
        "analysis_output_1126_for_pic/analysis_output_1126_12_4_4/well_comparisons",
        "analysis_output_1126_for_pic/analysis_output_1126_8_124_new/well_comparisons",
        "analysis_output_1126_for_pic/analysis_output_1126_4812_1_new/well_comparisons",
    ]

    # Strings to search for in filenames
    # search_terms = ["苏21", "苏52-8", "苏14-7-53", "苏14-5-48", "苏14-8-01", "苏14-6-30C1", \
    #                 "鄂9", "苏59-8-53", "苏59-8-40X", "苏59-6-53A", "苏59-8-54AX3", "苏59-9-28X1", \
    #                 "苏48-18-38", "苏48-17-26", "苏48-20-40", "苏48-7-39", "苏48-17-24", "苏48-7-26"    ]

    # search_terms = [
    #     "苏56-8",
    #     "苏52-8",
    #     "苏55-6",
    #     "苏14-5-34",
    #     "苏14-9-36",
    #     "苏14-8-45",
    #     "苏14-5-41",
    #     "苏平14-13-39",
    #     "苏14-6-46"
    # ]
    search_terms = [
        "苏21",
        "苏52-8",
        "苏55-6",
        "苏14-5-34",
        "苏14-9-36",
        "苏14-8-45",
        "苏14-5-41",
        "苏14-7-18C3",
        "苏14-6-46"
        ]
    # Run the function
    pick_figures(folders_to_search, search_terms)

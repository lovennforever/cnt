# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 20:01:44 2024

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 09:39:42 2024

@author: Administrator
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import re

def DealWithTensionFile(base_data_path):

# Target timesteps
    target_timesteps = [0, 20000, 40000, 60000, 80000]
    
    def extract_timesteps_from_file(file_path, timesteps):
        with open(file_path, 'r') as file:
            content = file.read()
    
        # Find positions of all ITEM: TIMESTEP
        timestep_pattern = re.compile(r'(ITEM: TIMESTEP\s*\n\d+)', re.MULTILINE)
        matches = list(timestep_pattern.finditer(content))
    
        # Extract ITEM: TIMESTEP and the corresponding data
        timestep_data = {}
        for i, match in enumerate(matches):
            start_idx = match.start()
            next_start_idx = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            timestep_content = content[start_idx:next_start_idx]
    
            # Extract timestep number
            timestep_number = int(re.search(r'ITEM: TIMESTEP\s*\n(\d+)', timestep_content).group(1))
            timestep_data[timestep_number] = timestep_content
    
        # Process each target timestep
        for timestep in timesteps:
            if timestep in timestep_data:
                # Construct output file name
                output_file_path = os.path.join(os.path.dirname(file_path), f'tension_{timestep}.xyz')
                with open(output_file_path, 'w') as file:
                    file.write(timestep_data[timestep])
                print(f'Saved data for timestep {timestep} to file {output_file_path}')
            else:
                print(f'Timestep {timestep} not found in file {file_path}')
    
    def process_directory(directory_path):
        # Traverse directory and subdirectories
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('tension.xyz'):
                    file_path = os.path.join(root,file)
                    print(f'Processing file: {file_path}')
                    extract_timesteps_from_file(file_path, target_timesteps)
    
    if __name__ == "__main__":
        process_directory(base_data_path)
        
    

def CntDensityData(base_data_path):
    
    # Dictionary to store min values and corresponding x-axis
    min_values = {}
    
    # Walk through main folder and its subdirectories
    for root, dirs, files in os.walk(base_data_path):
        # Get all tesion.xyz files in the current directory
        xyz_files = glob.glob(os.path.join(root, 'tension_*.xyz'))  
        if not xyz_files:
            continue
        plt.figure(figsize=(12, 8))
        plot_data = {}
        min_y_80000 = float('inf')
        min_x_80000 = None
    
        # Process each tension file
        for file_path in xyz_files:
            file_name = os.path.basename(file_path)
    
            # Extract number from the tension file name
            match = re.search(r'\d+', file_name)#Find the first continuous number sequence
            label = match.group()[:2] if match else 'unknown'
    
            # Read the data, ignoring the first 9 rows
            data = pd.read_csv(file_path, delim_whitespace=True, header=None, skiprows=9)
            data.columns = [f'col{i+1}' for i in range(data.shape[1])]
    
            # Divide the fourth column into 10 quantiles
            data['quartile'] = pd.cut(data['col4'], bins=10, labels=False, include_lowest=True)
    
            # Count occurrences of col1 values in each quartile
            result = data[data['col1'].isin([3])].groupby('quartile').size().reindex(range(10), fill_value=0)#number 3 represent CNT atom of zigzag structure
    
            # Store current plot data
            plot_data[label] = result.values
    
            plt.plot(result.index + 1, result.values, marker='o', linestyle='-', label=f'{label} ps', markersize=8, linewidth=2)
            # Check if filename is *tension_80000.xyz
            if 'tension_80000' in file_name:
                ydata = result.values
                xdata = result.index + 1
                min_y_in_line = min(ydata)
                if min_y_in_line < min_y_80000:
                    min_y_80000 = min_y_in_line
                    min_x_80000 = xdata[ydata.argmin()]

    
            # Record the min value of CNT density and corresponding x value
            min_y = result.min()
            min_x = result.idxmin() + 1
            min_values[label] = (min_x, min_y)  # Store in the dictionary
    
            # Save quartile data to a text file
            for quartile, group in data.groupby('quartile'):
                selected_data = group.iloc[:, :4]
                output_filename = os.path.join(root, f'{file_name}_quartile_{quartile}.txt')
                selected_data.to_csv(output_filename, index=False, header=False, sep='\t', lineterminator='\n\n')
    
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
            spine.set_linestyle('-')
    
        # Draw vertical line for fracture position
        if min_x_80000 is not None:
            plt.axvline(x=min_x_80000, color='red', linestyle='--', linewidth=2, label='Fracture position')
            plt.scatter(min_x_80000, min_y_80000, color='red', zorder=5)
            plt.text(min_x_80000, min_y_80000, f'{min_y_80000:.2f}', color='red', fontsize=20, verticalalignment='bottom', horizontalalignment='right')
    
            # Save fracture position to file
            fracture_position_file_path = os.path.join(root, 'fracture_position_c60.txt')
            with open(fracture_position_file_path, 'w') as f:
                f.write(f'Fracture position (x) for tension_80000.xyz: {min_x_80000}\n')
    
        # Set plot labels and styles
        plt.xlabel('N', fontsize=20)
        plt.ylabel('Atoms', fontsize=20)
        plt.xticks(range(1, 11))
        plt.legend(loc='best', prop={'size': 20})
        plt.tick_params(axis='both', which='major', labelsize=16)
    
        # Create DataFrame for plot data
        plot_df = pd.DataFrame(index=range(1, 11), columns=sorted(plot_data.keys()))
        for label, values in plot_data.items():
            plot_df[label] = values
    
        # Save plot data to a CSV file
        data_output_path = os.path.join(root, 'c60.csv')
        plot_df.to_csv(data_output_path)
    
        # Save min values to file
        min_values_output_path = os.path.join(root, 'min_values_c60.txt')
        with open(min_values_output_path, 'w') as f:
            for label, (min_x, min_y) in min_values.items():
                f.write(f'{label} Min y = {min_y}, at x = {min_x}\n')
    
        # Save the plot image
        output_image_path = os.path.join(root, 'c60.png')
        plt.savefig(output_image_path, bbox_inches='tight', dpi=600)
    
        # Close the plot for the next iteration
        plt.close()
        
  
def C60DensityData(base_data_path):
    
    # Dictionary to store min values and corresponding x-axis
    min_values = {}
    
    # Walk through main folder and its subdirectories
    for root, dirs, files in os.walk(base_data_path):
        # Get all tesion.xyz files in the current directory
        xyz_files = glob.glob(os.path.join(root, 'tension_*.xyz'))  
        if not xyz_files:
            continue
        plt.figure(figsize=(12, 8))
        
        plot_data = {}
    
        # Process each tension file
        for file_path in xyz_files:
            file_name = os.path.basename(file_path)
    
            # Extract number from the tension file name
            match = re.search(r'\d+', file_name)#Find the first continuous number sequence
            label = match.group()[:2] if match else 'unknown'
    
            # Read the data, ignoring the first 9 rows
            data = pd.read_csv(file_path, delim_whitespace=True, header=None, skiprows=9)
            data.columns = [f'col{i+1}' for i in range(data.shape[1])]
    
            # Divide the fourth column into 10 quantiles
            data['quartile'] = pd.cut(data['col4'], bins=10, labels=False, include_lowest=True)
    
            # Count occurrences of col1 values in each quartile
            result = data[data['col1'].isin([1, 2])].groupby('quartile').size().reindex(range(10), fill_value=0)#number 1 and 2 represent C60 atom of zigzag structure
    
            # Store current plot data
            plot_data[label] = result.values
    
            plt.plot(result.index + 1, result.values, marker='o', linestyle='-', label=f'{label} ps', markersize=8, linewidth=2)
    
            # Record the min value of CNT density and corresponding x value
            min_y = result.min()
            min_x = result.idxmin() + 1
            min_values[label] = (min_x, min_y)  # Store in the dictionary
    
            # Save quartile data to a text file
            for quartile, group in data.groupby('quartile'):
                selected_data = group.iloc[:, :4]
                output_filename = os.path.join(root, f'{file_name}_quartile_{quartile}.txt')
                selected_data.to_csv(output_filename, index=False, header=False, sep='\t', lineterminator='\n\n')
    
        ax = plt.gca()
    
        # Set border styles
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
            spine.set_linestyle('-')

        plt.xlabel('N', fontsize=20)
        plt.ylabel('Atoms', fontsize=20)
        plt.xticks(range(1, 11))
        plt.legend(loc='best', prop={'size': 20})
        plt.tick_params(axis='both', which='major', labelsize=16)
    
        # Create an empty DataFrame for plot data
        plot_df = pd.DataFrame(index=range(1, 11), columns=sorted(plot_data.keys()))
        
        # Fill the DataFrame
        for label, values in plot_data.items():
            plot_df[label] = values

        plot_df.to_csv(os.path.join(root, 'c60.csv'))
        with open(os.path.join(root, 'density_min_values_c60.txt'), 'w') as f:
            for label, (min_x, min_y) in min_values.items():
                f.write(f'{label} Min y = {min_y}, at x = {min_x}\n')

        plt.savefig(os.path.join(root, 'c60.png'), bbox_inches='tight', dpi=600)
        
        # Close the plot for the next iteration
        plt.close()    
        
        

def C60DensityDataRelativeCoordinates(base_data_path):    

    for root, dirs, files in os.walk(base_data_path):
        for file in files:
            if file == 'c60.csv':
                # Construct the full path for 'c60.csv' and 'cnt.csv'
                plot_data_path = os.path.join(root, file)
                cnt_data_path = os.path.join(root, 'cnt.csv')
    
                # Read the cnt and c60 CSV files
                plot_data = pd.read_csv(plot_data_path)
                cnt_data = pd.read_csv(cnt_data_path)
    
                # Find the minimum value of the sixth column of the cnt.txt file, and get the    corresponding X value in the 1st column of the cnt.txt file
                min_value_index = cnt_data.iloc[:, 5].idxmin()
                min_value_x = cnt_data.iloc[min_value_index, 0]
    
                # Modify the first column of plot_data by subtracting min_value_x
                plot_data.iloc[:, 0] = plot_data.iloc[:, 0] - min_value_x

                modified_data_path = os.path.join(root, 'c60_modified.csv')
                plot_data.to_csv(modified_data_path, index=False)

                fig, ax = plt.subplots(figsize=(12, 8))

                marker_style = 'o'
                for col in plot_data.columns[1:]:
                    ax.plot(plot_data.iloc[:, 0], plot_data[col], label=f'{col} ps', marker=marker_style)

                ax.set_xlabel('N', fontsize=20)
                ax.set_ylabel('Atom', fontsize=20)
                ax.tick_params(axis='both', which='major', labelsize=20)

                ax.legend(fontsize=16, loc='upper left', bbox_to_anchor=(1.05, 0.95), frameon=True, edgecolor='black')
    
                # Set spine width
                for spine in ['left', 'bottom']:
                    ax.spines[spine].set_linewidth(1.5)

                plt.tight_layout()
                plt.savefig(os.path.join(root, 'c60_relative.png'), bbox_inches='tight')
                plt.close()

                print(f"All plots generated and modified data saved in directory: {root}")   
                
                
def ProcessDensityFiles(base_data_path):
    # Extract numbers from the density_min_values_c60.txt file
    def extract_numbers(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        results = []
        
        for line in lines:
            # Extract all numbers from the line
            numbers = re.findall(r'\d+', line)
            
            if len(numbers) >= 3:
                # Store the first three numbers as a tuple
                results.append(tuple(numbers[:3]))
        
        return results

    # Write extracted results to a new file
    def write_results_to_file(output_file_path, results):
        with open(output_file_path, 'w') as file:
            # Write header
            file.write("time density_min_c60 corresponding_x\n")
            # Write extracted values
            for values in results:
                file.write(f"{' '.join(values)}\n")

    # Process the file and select lines with minimum time difference
    def process_file(file_path):
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist!")
            return
    
        with open(file_path, 'r') as file:
            lines = file.readlines()
    
        if len(lines) < 5:
            print(f"File {file_path} has less than 5 lines, cannot process.")
            return
        
        lines = lines[1:]
        
        y_0_min = float(lines[0].split()[1])
        x_0 = float(lines[0].split()[0])
    
        selected_lines = []
        min_diff = float('inf')
    
        for line in lines:
            data = line.split()
            if len(data) < 2:
                continue
    
            y_value = float(data[1])
    
            if y_value < y_0_min:
                diff = abs(float(data[0]) - x_0)
                if diff <= min_diff:
                    if diff < min_diff:
                        selected_lines.clear()
                        min_diff = diff
                    selected_lines.append(data)
    
        if selected_lines:
            # Change name
            output_file_path = file_path.replace('.txt', '_min_time_line.txt')
            with open(output_file_path, 'w') as output_file:
                for line in selected_lines:
                    output_file.write(' '.join(line) + '\n')
    
            print(f"Data saved in {output_file_path}")
        else:
            print(f"In file {file_path} no suitable lines found.")

    # Process all files in directory
    def process_all_files_in_directory(directory):
        for root, dirs, files in os.walk(directory):
            if 'density_min_values_c60.txt' in files:
                input_file_path = os.path.join(root, 'density_min_values_c60.txt')
                output_file_path = os.path.join(root, 'density_min_values_c60_corresponding_x.txt')
                
                extracted_values = extract_numbers(input_file_path)
                write_results_to_file(output_file_path, extracted_values)
    
                print(f"Extraction completed. Results saved to {output_file_path}")
                
                # Process the newly created file
                process_file(output_file_path)

    process_all_files_in_directory(base_data_path)
    
def MergeC60CntMinXdata(base_data_path): 
    
    # Output file path
    output_file = os.path.join(base_data_path, 'merged_C60Cnt_Min_Xdata.txt')
    
    # Regex to extract numbers
    number_pattern = re.compile(r'-?\d+\.?\d*')
    
    # List to store merged data
    merged_data = []
    
    # Traverse subdirectories
    for root, dirs, files in os.walk(base_data_path):
        last_numbers = []
    
        for file in files:
            if file == 'density_min_values_c60_corresponding_x_min_time_line.txt':
                with open(os.path.join(root, file), 'r') as f:
                    content = f.read().strip().splitlines()
                    if content:
                        last_numbers.append(number_pattern.findall(content[-1])[-1])
    
            elif file == 'fracture_position_cnt.txt':
                with open(os.path.join(root, file), 'r') as f:
                    content = f.read()
                    last_numbers.append(number_pattern.findall(content)[-1])
    
        # Append if both files are found
        if len(last_numbers) == 2:
            merged_data.append(' '.join(last_numbers))
    
    # Save merged data to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(merged_data))
    
    print(f"Merging complete, output saved to: {output_file}")
    
    
def HistogramC60Cnt(base_data_path):
    
    fracture_positions_file=os.path.join(base_data_path,'merged_C60Cnt_Min_Xdata.txt')
    # Read data from the first column of the file
    fracture_positions_df = pd.read_csv(fracture_positions_file, sep='\s+', header=None)
    x_values = fracture_positions_df.iloc[:, 0]
    y_values = fracture_positions_df.iloc[:, 1]
    
    # Calculate the difference
    diff_values = x_values - y_values
    
    # Calculate the histogram of differences and frequency counts
    counts, bin_edges = np.histogram(diff_values, bins=10)
    
    # Calculate bin centers and frequency medians
    bin_centers = []
    freq_medians = []
    
    for i in range(len(bin_edges) - 1):
        # Get data within the current bin
        bin_data = diff_values[(diff_values >= bin_edges[i]) & (diff_values < bin_edges[i + 1])]
        if len(bin_data) > 0:
            # Use frequency if greater than 0
            freq_median = counts[i] if counts[i] > 0 else 0
            freq_medians.append(freq_median)
    
            # Calculate the center of the bin
            bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
            bin_centers.append(bin_center)
    
    plt.figure(figsize=(10, 6))
    
    # Plot frequency medians with points and connect them
    plt.scatter(bin_centers, freq_medians, color='r', s=100, label='Frequency Medians')
    plt.plot(bin_centers, freq_medians, color='r', linestyle='dashed', linewidth=2)
    
    plt.xlabel('ΔN', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)

    save_path = os.path.join(base_data_path,'H.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=600)
    
    plt.legend()
    plt.show()
    
    # Print frequency medians for each bin
    print("Frequency medians:", freq_medians)   
    
    
base_data_path = r'./data/avg_se'

 
#DealWithTensionFile(base_data_path) 
#CntDensityData(base_data_path) 
#C60DensityData(base_data_path)
#C60DensityDataRelativeCoordinates(base_data_path)
#ProcessDensityFiles(base_data_path)
#MergeC60CntMinXdata(base_data_path)
#HistogramC60Cnt(base_data_path)
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 09:39:42 2024

@author: Administrator
"""

import os
import re
def DealWithTensionFile(main_directory_path):

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
                if file.endswith('.xyz'):
                    file_path = os.path.join(root, file)
                    print(f'Processing file: {file_path}')
                    extract_timesteps_from_file(file_path, target_timesteps)
    
    if __name__ == "__main__":
        process_directory(main_directory_path)
    
main_directory_path = r'./data/avg_se'    
#DealWithTensionFile(main_directory_path)  

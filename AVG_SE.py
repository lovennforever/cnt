#AVG_SE
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


def CntMinValue(base_data_path):
    
    for root, dirs, files in os.walk(base_data_path):
        if 'cnt.csv' in files:
            file_path = os.path.join(root, 'cnt.csv')
            df = pd.read_csv(file_path)           
            col_6 = df.iloc[:, 5]
            #The number of cnt atoms corresponding to each block of 80 ps
            min_value_col6 = col_6.min()
            min_index = col_6.idxmin()
            corresponding_value = df.iloc[min_index, 0]
            # print(f"Processing file: {file_path}")
            # print("min_value= ", min_value_col6)
            # print("cnt_frcture point= ", corresponding_value)
            save_path = os.path.join(root, "min_value.txt")
            with open(save_path, 'w') as f:
                f.write(f"min_value= {min_value_col6}\n")
                f.write(f"cnt_frcture point= {corresponding_value}")
            
            print("Value saved to: ", save_path)

            
def C60DataChange(base_data_path):

    for root, dirs, files in os.walk(base_data_path):
        for curDir in dirs:
            min_value_file_path = os.path.join(root, curDir, "min_value.txt")
            csv_file_path = os.path.join(root, curDir, "c60.csv")
            print("min_value_file_path:", min_value_file_path)
            print("csv_file_path:", csv_file_path)
            try:
                with open(min_value_file_path, 'r') as file:
                    min_value = float(file.readlines()[1].split("=")[-1].strip())
            except Exception as e:
                print(f"Error reading {min_value_file_path}: {e}")
                continue            
            try:
                df_c60 = pd.read_csv(csv_file_path)
            except Exception as e:
                print(f"Error reading {csv_file_path}: {e}")
                continue
            #The first column of C60.CSV minus the second line of min_value.txt
            df_c60.iloc[:, 0] -= min_value
            
            output_file_path = os.path.join(root, curDir, "c60_modified.csv")
            try:
                df_c60.to_csv(output_file_path, index=False)
                print(f"Modified data saved to {output_file_path}")
            except Exception as e:
                print(f"Error saving file: {e}")
                
                
def CurveC60DensityRelativeCoordinates(base_data_path): 
               
    for root, dirs, files in os.walk(base_data_path):
    # Read the min_value.txt file
        for curDir in dirs:
            modified_csv_file_path = os.path.join(root, curDir, "c60_modified.csv")
        
            if not os.path.exists(modified_csv_file_path):
                print(f"File {modified_csv_file_path} does not exist.")
                exit()

            # Read the c60_modified.csv file
            try:
                df_modified = pd.read_csv(modified_csv_file_path)
            except Exception as e:
                print(f"Error reading modified CSV file: {e}")
                exit()
            if df_modified.shape[1] < 2:
                print("The c60_modified.csv file does not contain enough data.")
                exit()
    
            # Extract the x-axis data (first column)
            x_data = df_modified.iloc[:, 0]

            plt.figure(figsize=(10, 6))
            for i in range(1, df_modified.shape[1]):
                y_data = df_modified.iloc[:, i]
                label = f"{20 * i}"  # Generate label  
                plt.plot(x_data, y_data, label=label)  # Plot
            plt.legend(bbox_to_anchor=(1.05, 0.5), loc='upper left')
            plt.xlabel('ΔN')
            plt.ylabel('Atom')
            save_path = os.path.join(root, curDir, f"{curDir}_c60_plot.png")
            print(save_path)
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            plt.tight_layout()
            plt.show()
            
            
def CalAvgSeData(base_data_path):

    dataframes = []
    
    for root, dirs, files in os.walk(base_data_path):
        for curDir in dirs:
            folder = os.path.join(root, curDir)
            print(f"Processing folder: {folder}")
            file_name = 'c60_modified.csv'
            file_path = os.path.join(folder, file_name)

            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                dataframes.append(df)
            else:
                print(f"File not found: {file_path}")

    # Columns to calculate averages and standard errors for
    columns_to_average = ['Unnamed: 0', '0', '20', '40', '60', '80']

    avg_df = pd.DataFrame()
    std_err_df = pd.DataFrame()
    
    # Calculate average and standard error for each column
    for column in columns_to_average:
        print(f"--- Calculating for column: {column} ---")
        
        # Calculate average
        avg_values = np.array([df[column].values for df in dataframes])
        avg_value = np.mean(avg_values, axis=0)
        avg_df[column] = avg_value
        print(f"Column: {column}, Average: {avg_value}")
        
        # Calculate standard error
        std_dev = np.std(avg_values, axis=0, ddof=1)  # Sample standard deviation
        std_err = std_dev / np.sqrt(len(dataframes))
        std_err_df[column] = std_err
        print(f"Column: {column}, Standard Error: {std_err}")
    
    # Save the results to Excel files
    output_AVG_file_path = os.path.join(base_data_path, 'AVG.xlsx')
    output_SE_file_path = os.path.join(base_data_path, 'SE.xlsx')
    
    with pd.ExcelWriter(output_AVG_file_path) as writer:
        avg_df.to_excel(writer, sheet_name='Average', index=False)
    
    with pd.ExcelWriter(output_SE_file_path) as writer:
        std_err_df.to_excel(writer, sheet_name='Standard Error', index=False)
    
    print(f"Averages and standard errors have been calculated and saved to: {output_AVG_file_path}")
    print(f"Averages and standard errors have been calculated and saved to: {output_SE_file_path}")            


def CurveAvgSe(base_data_path):
    
    # Define file paths
    avg_file_path = os.path.join(base_data_path,'AVG.xlsx')
    se_file_path =os.path.join(base_data_path,'SE.xlsx') 
    
    # Read average and standard error data from Excel files
    avg_df = pd.read_excel(avg_file_path)
    se_df = pd.read_excel(se_file_path)

    # Get x values (first column data) and y values (other columns)
    x_values = avg_df.iloc[:, 0]
    y_values = avg_df.iloc[:, 1:]
    se_values = se_df.iloc[:, 1:]

    plt.figure(figsize=(10, 6))
    
    # Plot each column with error bands
    for i, column in enumerate(y_values.columns):
        plt.plot(x_values, y_values[column], label=f"AVG {column} ps", linewidth=2)
        plt.fill_between(x_values, 
                         y_values[column] - se_values.iloc[:, i],
                         y_values[column] + se_values.iloc[:, i], 
                         alpha=0.2, label=f'SE {column} ps')

    plt.xlabel('ΔN', fontsize=20)
    plt.ylabel('Atom', fontsize=20)
    plt.legend(bbox_to_anchor=(1.2, 0.5), loc='center', fontsize=20)
    # Adjust tick parameters
    ax = plt.gca()  # Get current axes object
    ax.tick_params(axis='both', which='major', labelsize=16)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)  # Set border width to 1.5
    file_path = os.path.join(base_data_path, 'AVG_SE_plot.png')

    plt.savefig(file_path,dpi=600, bbox_inches='tight')

    plt.show()
    
    
base_data_path = r"./data/avg_se"

#CntMinValue(base_data_path)
#C60DataChange(base_data_path)
#CurveC60DensityRelativeCoordinates(base_data_path)
#CalAvgSe(base_data_path)
#CurveAvgSe(base_data_path)
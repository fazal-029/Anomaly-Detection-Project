import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

def load_data(txt_files, folder_path, is_record, fraction_of_anomaly, mutation_done = True):
    dataset = []
    pattern = r'_(\d+)_(\d+)_(\d+)\.txt$'
    count_of_matching_files = 0
    for txt_file in txt_files:
        match = re.search(pattern, txt_file)
        if match:
            last_training_data = int(match.group(1))
            begin_anomaly = int(match.group(2))
            end_anomaly = int(match.group(3))
            # print(f"File: {txt_file}, Numbers: {last_training_data}, {begin_anomaly}, {end_anomaly}")
        
            #I am reading the values from the .txt file
            values = []
            file_path = folder_path + txt_file
                
            with open(file_path, 'r') as file:
                for line in file:
                    # Split the line into individual strings
                    parts = line.strip().split()
                    for part in parts:
                        try:
                            values.append(float(part))
                        except ValueError:
                            print(f"Skipping invalid value: {part}")
                                
            df = pd.DataFrame(values, columns=['feature'])
        
            
                    
            if mutation_done is True:
                # My convention: 0 means normal data and 1 means an anomaly
                df['is_anomaly'] = 0
                for each in range(begin_anomaly, end_anomaly+1):
                    df.loc[df.index == each, 'is_anomaly'] = 1
                    
                if is_record:
                    noisy_df = mutation(df,last_training_data,record = True, fraction_of_anomaly = fraction_of_anomaly)
                    dataset.append((noisy_df,last_training_data, begin_anomaly, end_anomaly))
                else:
                    noisy_df = mutation(df,last_training_data,record = False, fraction_of_anomaly = fraction_of_anomaly)
                    dataset.append((noisy_df,last_training_data, begin_anomaly, end_anomaly))
            else:
                dataset.append((df, last_training_data, begin_anomaly, end_anomaly))
            count_of_matching_files = count_of_matching_files + 1
        else:
            print(f"No match found in file: {txt_file}")

    print(f'Number of matching files: {count_of_matching_files}')
    print('One sample dataframe: ')
    print(dataset[0])
    return dataset

def add_noise(original_time_series, last_training_data, noise_level, fraction_of_anomaly): # anomolous record
    # print("Adding noise")
    time_series = original_time_series.copy()
    # Number of points to modify
    n_points = int(len(time_series[:last_training_data]) * fraction_of_anomaly)
    
    # Randomly selected indices to modify
    random_indices = np.random.choice(time_series[:last_training_data].index, size=n_points, replace=False)
    
    # Adding noise to the selected points
    noise = noise_level * np.mean(time_series['feature'].iloc[:last_training_data])
    time_series.loc[random_indices,'feature'] += noise
    time_series.loc[random_indices,'is_anomaly'] = 1
    
    return time_series

def horizontal_shift(original_time_series, last_training_data, shift_amount = 15, fraction_of_anomaly = 0.01): # anomolous sequence
    # print("Horizontal shift")
    time_series = original_time_series.copy()
    # Number of points to modify
    n_points = int(len(time_series[:last_training_data]) * fraction_of_anomaly) 
    
    # Randomly selecting the start index
    start_index = np.random.randint(1, len(time_series[:last_training_data]) - n_points)
    prev_value = time_series.at[start_index-1, 'feature']
    
    # Ensuring that the shift does not go out of bounds
    end_index = start_index + n_points
    if end_index + shift_amount >= len(time_series[:last_training_data]):
        shift_amount = len(time_series[:last_training_data]) - end_index - 1

    # Create a shifted version of the selected segment
    shifted_segment = time_series.loc[start_index:end_index, 'feature'].values
    time_series.loc[start_index + shift_amount:end_index + shift_amount, 'feature'] = shifted_segment
    
    # Filling the gap with the previous value
    time_series.loc[start_index:start_index + shift_amount, 'feature'] = prev_value
    
    # Marking the anomalous segment
    time_series.loc[start_index:end_index + shift_amount, 'is_anomaly'] = 1
    return time_series

def vertical_shift(original_time_series,last_training_data, fraction_of_anomaly = 0.01): # anomolous sequence
    # print("Vertical shift")
    time_series = original_time_series.copy()
    n_points = int(len(time_series[:last_training_data]) * fraction_of_anomaly)
    
    # Randomly selecting the start index
    start_index = np.random.randint(0, len(time_series[:last_training_data]) - n_points)
    end_index = start_index + n_points
    
    # Calculate the random shift value between the min and max of the feature
    min_value = time_series['feature'].iloc[:last_training_data].min()
    max_value = time_series['feature'].iloc[:last_training_data].max()
    random_shift = np.random.uniform(min_value, max_value)
    
    
    # Apply the vertical shift
    time_series.loc[start_index:end_index, 'feature'] += random_shift
    
    # Marking the anomalous segment
    time_series.loc[start_index:end_index, 'is_anomaly'] = 1
    
    return time_series

def rescale(original_time_series, last_training_data, factor = 3, fraction_of_anomaly = 0.01): # anomolous sequence
    # print("rescale")
    time_series = original_time_series.copy()
    n_points = int(len(time_series[:last_training_data]) * fraction_of_anomaly)  # 1% of the time series data
    
    # Randomly selecting the start index
    start_index = np.random.randint(0, len(time_series[:last_training_data]) - n_points)
    end_index = start_index + n_points
    
    # Apply rescale
    time_series.loc[start_index:end_index, 'feature'] *= factor
    
    # Marking the anomalous segment
    time_series.loc[start_index:end_index, 'is_anomaly'] = 1
    
    return time_series

def add_dense_noise(original_time_series, last_training_data, fraction_of_anomaly = 0.01):
    # print("adding dense noise")
    time_series = original_time_series.copy()
    n_points = int(len(time_series[:last_training_data]) * fraction_of_anomaly)  # Modify 1% of the time series data
    
    # Randomly selecting the start index for the noise
    start_index = np.random.randint(0, len(time_series[:last_training_data]) - n_points)
    
    # Generating random noise values
    random_values = np.random.randn(n_points)
    
    # Replacing the selected segment with random values
    time_series.loc[start_index:start_index + n_points - 1, 'feature'] = random_values * 1000
    
    # Marking the noisy segment as an anomaly
    time_series.loc[start_index:start_index + n_points - 1, 'is_anomaly'] = 1
    
    return time_series
    
    

def mutation(time_series, last_training_data, record = False, fraction_of_anomaly = 0.01):
    # mutation operator will be randomly chosen
    mutation_type = np.random.randint(1,5)
    if record: # adding noise introuduces anomolous records
        noisy_time_series = add_noise(time_series, last_training_data, noise_level = 20, fraction_of_anomaly = fraction_of_anomaly)
    else: # other mutation operators introuduce anomolous sequences
        if mutation_type == 1:
            noisy_time_series = horizontal_shift(time_series, last_training_data, shift_amount = 15, fraction_of_anomaly = 0.02)
        elif mutation_type == 2:
            noisy_time_series = vertical_shift(time_series, last_training_data, fraction_of_anomaly = 0.02)
        elif mutation_type == 3:
            noisy_time_series = rescale(time_series, last_training_data,factor = 3, fraction_of_anomaly = 0.02)
        elif mutation_type == 4:
            noisy_time_series = add_dense_noise(time_series, last_training_data, fraction_of_anomaly = 0.02)
    
    return noisy_time_series

def calculate_statistics(data):
    mean = np.mean(data['feature'].iloc[:last_training_data])
    variance = np.var(data['feature'].iloc[:last_training_data])
    print(f"For training portion: Mean: {mean}, Variance: {variance}")

def anomaly_count(df):
    count_anomalies = df[df['is_anomaly'] == 1].shape[0]
    print("Number of 1s in 'is_anomaly' column:", count_anomalies)


def plot_the_time_series(df, noisy_df, last_training_data):
    df = df.loc[:last_training_data]
    noisy_df = noisy_df.loc[:last_training_data]
    plt.rcdefaults()
    plt.style.use('default')
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['feature'], label='Original time series', color='blue')
    plt.plot(noisy_df['feature'], label='After adding noise ', color='red', alpha=0.7)
    plt.legend()
    plt.show()
    
import numpy as np
import pandas as pd
import os
import re
from mutation import *

def load_data(txt_files, folder_path, is_record = False, fraction_of_anomaly = 0.02, mutation_done = True):
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

def custom_specificity(y_true, y_pred, tolerance=100):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    total_predictions = len(y_pred)
    
    tolerance = max(tolerance, 100)

    for i in range(total_predictions):
        if y_pred[i] == 1:
            # Check if there is any anomaly in the ground truth within the range p-100 to p+100
            if np.any(y_true[max(0, i - tolerance):min(len(y_true), i + tolerance + 1)] == 1):
                tp += 1
            else:
                fp += 1
        elif y_pred[i] == 0 and y_true[i] == 0:
            tn += 1
        elif y_pred[i] == 0 and y_true[i] == 1:
            fn += 1

    specificity = tn / (tn + fp) if (tn+fp) > 0 else 0.0
    return specificity
    
# Define the custom accuracy function
def is_prediction_correct(prediction, begin, end):
    print(f'prediction: {prediction}, begin: {begin}, end: {end}')
    L = end - begin + 1
    return min(begin - L, begin - 100) < prediction < max(end + L, end + 100)

def create_sequences_w_labels(data, labels, window_size):
    X = []
    y = []
    for i in range(len(data) - window_size + 1):
        X.append(data[i:(i + window_size)])
        segment = labels[i:(i + window_size)]
        if 1 in segment:
            y.append([1])
        else:
            y.append([0])
    return np.array(X), np.array(y)

def create_sequences_wo_labels(data, timesteps):
    sequences = []
    for i in range(len(data) - timesteps + 1):
        seq = data[i:i + timesteps]
        sequences.append(seq)
    return np.array(sequences)


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
    
import pandas as pd
import os
import re
import numpy as np

def find_correct_key(item):
    if any(substring in item for substring in ["Sel", "ECG", "ecg", "sdd", "BIDMC", "s2010", "ltstdbs30791"]):
        return "ecg"
    elif "taichi" in item:
        return "taichi"
    elif any(substring in item for substring in ["Marker", "gait", "Aceleration", "weallwalk", "park"]):
        return "walking"
    elif "resperation" in item:
        return "respiration"
    elif "tilt" in item:
        return "apb"
    elif "AirTemperature" in item:
        return "airtemp"
    elif "InternalBleeding" in item:
        return "bp"
    elif "EPG" in item:
        return "epg"
    elif "Densirostris" in item:
        return "whale"
    elif "powerdemand" in item or "PowerDemand" in item:
        return "power"
    elif "MARS" in item:
        return "nasa"
    else:
        return "skip"

# first list will contain values, second list will have tuples(begin_anomaly, end_anomaly), third list contains last training points
def initialize():
    dataset_dict = {"ecg":([],[], []), "taichi":([],[], []), "walking":([],[], []), "respiration":([],[], []), 
               "apb":([],[], []), "airtemp":([],[], []), "bp":([],[], []), "epg":([],[], []), "whale":([],[], []),
               "power":([],[], []), "nasa":([],[], [])}
    return dataset_dict

def load_data(txt_files, folder_path):
    dataset_dict = initialize()
    dataset = []
    pattern = r'_(\d+)_(\d+)_(\d+)\.txt$'
    count_of_matching_files = 0
    for txt_file in txt_files:
        number_of_samples_till_now = 0
        match = re.search(pattern, txt_file)
        if match:
            last_training_data = int(match.group(1))
            begin_anomaly = int(match.group(2))
            end_anomaly = int(match.group(3))
            
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
                                
            key = find_correct_key(txt_file)
            if key == "skip":
                continue
            # print(txt_file)
            # print(key)
            number_of_samples_till_now = len(dataset_dict[key][0])
            dataset_dict[key][0].extend(values)
            dataset_dict[key][1].append((begin_anomaly+number_of_samples_till_now, end_anomaly+number_of_samples_till_now))
            dataset_dict[key][2].append((last_training_data+number_of_samples_till_now, number_of_samples_till_now+len(values)))
            count_of_matching_files += 1
        else:
            print(f"No match found in file: {txt_file}")
    
    print(f'considering {count_of_matching_files} many files')
            
    for each_key in dataset_dict:
        print('Now working with category: ' + each_key)
        df = pd.DataFrame(dataset_dict[each_key][0], columns=['feature'])
        df['is_anomaly'] = 0
        for bg, ed in dataset_dict[each_key][1]:
            for e in range(bg, ed+1):
                df.loc[df.index == e, 'is_anomaly'] = 1
        
        train_df = pd.DataFrame(columns=['feature', 'is_anomaly'])
        test_set = []
        prev = 0
        for i, tup in enumerate(dataset_dict[each_key][2]):
            ltd, length_of_d = tup
            if train_df.empty:
                train_df = df.iloc[prev:ltd]
            else:
                train_df = pd.concat([train_df, df.iloc[prev:ltd]], ignore_index=True)
            test_df = df.iloc[ltd:length_of_d]
            bg_anomaly = dataset_dict[each_key][1][i][0] - prev
            ed_anomaly = dataset_dict[each_key][1][i][1] - prev
            prev = length_of_d
            test_set.append((test_df, bg_anomaly, ed_anomaly))

        dataset.append((train_df, test_set))

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

def create_sequences(data, timesteps):
    sequences = []
    for i in range(len(data) - timesteps + 1):
        sequence = data[i:i + timesteps]
        sequences.append(sequence)
    return np.array(sequences)
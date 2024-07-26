import numpy as np
import matplotlib.pyplot as plt

def add_noise(original_time_series, last_training_data, noise_level, fraction_of_anomaly): # anomolous record
    # print("Adding noise")
    time_series = original_time_series.copy()
    # Number of points to modify
    n_points = int(len(time_series[:last_training_data]) * fraction_of_anomaly)
    np.random.seed(29)
    # Randomly selected indices to modify
    random_indices = np.random.choice(time_series[:last_training_data].index, size=n_points, replace=False)
    
    # Adding noise to the selected points
    noise = noise_level * np.max(time_series['feature'].iloc[:last_training_data])
    time_series.loc[random_indices,'feature'] += noise
    time_series.loc[random_indices,'is_anomaly'] = 1
    
    return time_series

def horizontal_shift(original_time_series, last_training_data, shift_amount = 15, fraction_of_anomaly = 0.01): # anomolous sequence
    # print("Horizontal shift")
    time_series = original_time_series.copy()
    # Number of points to modify
    n_points = int(len(time_series[:last_training_data]) * fraction_of_anomaly) 
    
    np.random.seed(39)
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
    np.random.seed(25)
    # Randomly selecting the start index
    start_index = np.random.randint(0, len(time_series[:last_training_data]) - n_points)
    end_index = start_index + n_points
    
    # Calculate the random shift value between the min and max of the feature
    # min_value = time_series['feature'].iloc[:last_training_data].min()
    max_value = time_series['feature'].iloc[:last_training_data].max()
    random_shift = max_value  # np.random.uniform(min_value, max_value)
    
    
    # Apply the vertical shift
    time_series.loc[start_index:end_index, 'feature'] += random_shift
    
    # Marking the anomalous segment
    time_series.loc[start_index:end_index, 'is_anomaly'] = 1
    
    return time_series

def rescale(original_time_series, last_training_data, factor = 3, fraction_of_anomaly = 0.01): # anomolous sequence
    # print("rescale")
    time_series = original_time_series.copy()
    n_points = int(len(time_series[:last_training_data]) * fraction_of_anomaly) 
    np.random.seed(35)
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
    np.random.seed(42)
    # Randomly selecting the start index for the noise
    start_index = np.random.randint(0, len(time_series[:last_training_data]) - n_points)
    
    # Generating random noise values
    random_values = np.random.randn(n_points)
    
    # Replacing the selected segment with random values
    time_series.loc[start_index:start_index + n_points - 1, 'feature'] = random_values * 1000
    
    # Marking the noisy segment as an anomaly
    time_series.loc[start_index:start_index + n_points - 1, 'is_anomaly'] = 1
    
    return time_series
    
    

def mutation(time_series, last_training_data, record = False, fraction_of_anomaly = 0.02):
    # mutation operator will be randomly chosen
    mutation_type = np.random.randint(1,5)
    if record: # adding noise introuduces anomolous records
        noisy_time_series = add_noise(time_series, last_training_data, noise_level = 10, fraction_of_anomaly = fraction_of_anomaly)
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
    
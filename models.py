from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Model, Sequential
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tensorflow.keras.layers import Input, Reshape, LSTM, RepeatVector, TimeDistributed, Dense, Dropout, Bidirectional, GRU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras import layers, models, Model
from sklearn.svm import SVC
from utility_functions import create_sequences_w_labels, is_prediction_correct


def mlp2(window_size = 30):

    # Model definition - Assuming create_sequences_w_labels returns a 2D array
    model = Sequential()
    model.add(Input(shape=(window_size,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

def lstm(window_size = 30):
    inputs = Input(shape=(window_size, 1))
    x = LSTM(128, activation='relu', return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(64, activation='relu', return_sequences=False)(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(16, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(), loss='binary_crossentropy')
    return model

def create_lstm_autoencoder(input_shape):
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='relu', return_sequences=False),
        RepeatVector(input_shape[0]),
        LSTM(32, activation='relu', return_sequences=True),
        Dropout(0.2),
        LSTM(64, activation='relu', return_sequences=True),
        Dropout(0.2),
        TimeDistributed(Dense(input_shape[1]))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def detect_anomaly(model, test_data, segment_length):
    predictions = model.predict(test_data)
    reconstruction_errors = np.mean(np.abs(predictions - test_data), axis=1)
    segment_errors = []

    # Calculate average error for each segment
    for i in range(len(reconstruction_errors) - segment_length + 1):
        segment_error = np.mean(reconstruction_errors[i:i + segment_length])
        segment_errors.append(segment_error)

    # Find the segment with the highest average error
    most_confident_anomalous_segment = np.argmax(segment_errors)
    center_of_anomalous_segment = most_confident_anomalous_segment + segment_length // 2

    return center_of_anomalous_segment

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def create_transformer_autoencoder(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, dropout):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(units=input_shape[-1])(x)
    outputs = x
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model


def bi_lstm(window_size=30):
    inputs = Input(shape=(window_size, 1))
    
    # Bidirectional LSTM layers
    x = Bidirectional(LSTM(128, activation='relu', return_sequences=True))(inputs)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(64, activation='relu', return_sequences=False))(x)
    x = Dropout(0.2)(x)
    
    # Dense layers
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(16, activation='relu')(x)
    
    # Output layer
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(), loss='binary_crossentropy')
    return model
    
def gru(window_size=30):
    inputs = Input(shape=(window_size, 1))
    x = GRU(128, activation='relu', return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = GRU(64, activation='relu', return_sequences=False)(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(16, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(), loss='binary_crossentropy')
    return model

def handle_each_time_series_svm(df_tuple, window_size=30, kernel='poly'):
    X = df_tuple[0]['feature'].values
    y = df_tuple[0]['is_anomaly'].values
    last_training_data = df_tuple[1]
    begin_anomaly = df_tuple[2]
    end_anomaly = df_tuple[3]
    
    scaler = MinMaxScaler()
    X[:last_training_data] = scaler.fit_transform(X[:last_training_data].reshape(-1, 1)).flatten()
    X[last_training_data:] = scaler.transform(X[last_training_data:].reshape(-1, 1)).flatten()

    X_sequences, y_sequences = create_sequences_w_labels(X, y, window_size)

    X_train, X_test = X_sequences[:last_training_data - window_size], X_sequences[last_training_data - window_size:]
    y_train, y_test = y_sequences[:last_training_data - window_size], y_sequences[last_training_data - window_size:]

    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    
    residuals = y_test - predictions
    anomaly_position = np.argmax(residuals)
    correct_or_not = is_prediction_correct(anomaly_position, begin_anomaly, end_anomaly)
    
    return correct_or_not

def evaluate_svm(dataset_seq, window_size=30, kernel='poly'):
    results = []
    for i, df in enumerate(dataset_seq):
        print(f'Training and evaluating on dataset {i + 1}')
        correct_or_not = handle_each_time_series_svm(df, window_size, kernel)
        results.append(correct_or_not)
        print(f'Dataset {i + 1} correctly identified---------------- {correct_or_not}')
    
    accuracy = sum(1 for item in results if item) / len(results)
    print(f'Final Accuracy: {accuracy}')
    return accuracy
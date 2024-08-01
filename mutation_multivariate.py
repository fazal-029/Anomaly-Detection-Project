import numpy as np
from sklearn.metrics import auc
from prts import ts_precision, ts_recall
from builtins import AssertionError
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import precision_recall_curve, auc

class AUC_PR_Callback(Callback):
    def __init__(self, validation_data, model_save_path):
        super().__init__()
        self.validation_data = validation_data
        self.model_save_path = model_save_path
        self.best_auc_pr = 0

    def on_epoch_end(self, epoch, logs=None):
        # Get the validation data
        val_x, val_y = self.validation_data

        # Predict on validation data
        val_predictions = self.model.predict(val_x).reshape(-1)

        current_auc_pr = get_auc_pr(val_predictions, val_y)

        # Compare with the best AUC-PR
        if current_auc_pr > self.best_auc_pr:
            self.best_auc_pr = current_auc_pr
            self.model.save_weights(self.model_save_path)
            print(f'\nEpoch {epoch + 1}: AUC-PR improved to {current_auc_pr:.4f}. Model weights saved.')


def get_precision_recall(real, pred):
    try:
        precision = ts_precision(real, pred, alpha=0.0, cardinality="reciprocal", bias="flat")
        recall = ts_recall(real, pred, alpha=0.0, cardinality="reciprocal", bias="flat")
    except AssertionError:
        precision = 0.0
        recall = 0.0
    return precision, recall

def get_auc_pr(test_predictions, true_values):
    precisions = []
    recalls = []
    residuals = np.abs(true_values - test_predictions)
    lowers = np.linspace(0.01, 0.3, num=50)
    uppers = np.linspace(0.99, 0.7, num=50)
    for l,u in zip(lowers, uppers):
        lower = np.quantile(residuals, l)
        upper = np.quantile(residuals, u)
        bin_pred = ((residuals > upper) | (residuals < lower)).astype(int)
        precision, recall = get_precision_recall(true_values, bin_pred)
        precisions.append(precision)
        recalls.append(recall)

    auc = auc_pr_helper(precisions, recalls)
    return auc

def auc_pr_helper(precisions, recalls):
    # Sort recall values in ascending order and adjust precisions accordingly
    sorted_indices = np.argsort(recalls)
    sorted_recalls = np.array(recalls)[sorted_indices]
    sorted_precisions = np.array(precisions)[sorted_indices]
    
    return auc(sorted_recalls, sorted_precisions)


def add_noise_multivariate(train_x_seq, train_y_seq, fraction_of_anomaly = 0.05, noise_level = 20): # for record
    # Number of points to modify
    n_points = int(train_x_seq.shape[0] * fraction_of_anomaly)
    np.random.seed(29)
    # Randomly selected indices to modify
    random_indices = np.random.choice(range(train_x_seq.shape[0]), size=n_points, replace=False)

    for ri in random_indices:
        noise = noise_level * np.max(train_x_seq[ri])
        train_x_seq[ri] = train_x_seq[ri] * noise
        train_y_seq[ri] = 1
    
    return train_x_seq, train_y_seq


def add_horizontal_shift_multivariate(train_x_seq, train_y_seq, fraction_of_anomaly = 0.05, shift_amount = 10):
    n_points = int(train_x_seq.shape[0] * fraction_of_anomaly)
    np.random.seed(30)
    start = np.random.randint(1, train_x_seq.shape[0] - 2*n_points)
    prev = train_x_seq[start - 1]

    for i in range(n_points):
        train_x_seq[start+i+shift_amount] = train_x_seq[start+i] 
        train_y_seq[start+i+shift_amount] = 1

    for j in range(shift_amount):
        train_x_seq[start+j] = prev
        train_y_seq[start+j] = 1 

    return train_x_seq, train_y_seq
    

def add_vertical_shift_multivariate(train_x_seq, train_y_seq, fraction_of_anomaly = 0.05):
    n_points = int(train_x_seq.shape[0] * fraction_of_anomaly)
    np.random.seed(31)
    start = np.random.randint(1, train_x_seq.shape[0] - 2*n_points)
    
    indices = np.argsort(train_x_seq.sum(axis = 1))
    noise = train_x_seq[indices[-1]]

    for i in range(n_points):
        train_x_seq[start+i] = noise 
        train_y_seq[start+i] = 1

    return train_x_seq, train_y_seq
    

def add_rescale_multivariate(train_x_seq, train_y_seq, fraction_of_anomaly = 0.05, scale = 20):
    # Number of points to modify
    n_points = int(train_x_seq.shape[0] * fraction_of_anomaly)
    np.random.seed(35)
    start = np.random.randint(1, train_x_seq.shape[0] - 2*n_points)

    for i in range(n_points):
        train_x_seq[start+i] = train_x_seq[start+i] * scale 
        train_y_seq[start+i] = 1

    return train_x_seq, train_y_seq


def add_dense_noise_multivariate(train_x_seq, train_y_seq, fraction_of_anomaly = 0.05):
    n_points = int(train_x_seq.shape[0] * fraction_of_anomaly)
    np.random.seed(42)
    start = np.random.randint(1, train_x_seq.shape[0] - 2*n_points)
    
    for i in range(n_points):
        train_x_seq[start+i] = np.random.rand(train_x_seq.shape[1]) * 100
        train_y_seq[start+i] = 1

    return train_x_seq, train_y_seq


def mutation_multivariate(train_x, train_y, fraction_of_anomaly = 0.05, record = False):
    # mutation operator will be randomly chosen
    mutation_type = np.random.randint(1,5)
    if record: # adding noise introuduces anomolous records
        train_x, train_y = add_noise_multivariate(train_x, train_y, fraction_of_anomaly = fraction_of_anomaly)
    else: # other mutation operators introuduce anomolous sequences
        if mutation_type == 1:
            train_x, train_y = add_horizontal_shift_multivariate(train_x, train_y, fraction_of_anomaly = fraction_of_anomaly)
        elif mutation_type == 2:
            train_x, train_y = add_vertical_shift_multivariate(train_x, train_y, fraction_of_anomaly = fraction_of_anomaly)
        elif mutation_type == 3:
            train_x, train_y = add_rescale_multivariate(train_x, train_y, fraction_of_anomaly = fraction_of_anomaly)
        elif mutation_type == 4:
            train_x, train_y = add_dense_noise_multivariate(train_x, train_y, fraction_of_anomaly = fraction_of_anomaly)
    
    return train_x, train_y

def load_multivariate_mutated(all_train_x, all_train_y, record = False):
    for k in range(all_train_x.shape[0]):
        all_train_x[k], all_train_y[k] = mutation_multivariate(all_train_x[k], all_train_y[k], fraction_of_anomaly = 0.05, record = record)

    return all_train_x, all_train_y
    
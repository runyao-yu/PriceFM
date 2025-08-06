import warnings
warnings.filterwarnings('ignore')

import os
import gc
import sys
import time
import numpy as np
import pandas as pd
import pickle
from collections import deque
from multiprocessing import Process

import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization, Permute, Dense, Input, Flatten, Add, Subtract, Layer,  Activation, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras import backend as K

import sklearn
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("Python version:", sys.version)
print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("Scikit-learn version:", sklearn.__version__)


def train_status(status):

    if status == "cloud":
        from google.colab import drive
        drive.mount('/content/drive')
        pre_path = "/content/drive/My Drive/PriceFM/"

    elif status == "local":
        pre_path = os.path.abspath(".") + "/"
        
    if not os.path.exists(pre_path):
        os.makedirs(pre_path)

    return pre_path


def filter_columns(df, features_to_keep):

    filtered_columns = [
            col for col in df.columns 
            if any(feature in col for feature in features_to_keep)
        ]
    filtered_columns = filtered_columns + ['CET']
    return df[filtered_columns]


def convert_datetime(time_string):
    return pd.to_datetime(time_string).strftime('%Y-%m-%d')


def create_daily_windows(x_scaled, y_scaled, times, x_offsets=(-72, -1), y_offsets=(0, 23)):

    samples_X = []
    samples_Y = []
    times = pd.to_datetime(times)
    n = len(times)

    # Calculate window lengths (inclusive of endpoints)
    y_window_length = y_offsets[1] - y_offsets[0] + 1

    # Iterate over indices looking for a marker (time == 23:00)
    for i in range(n):
        if times.iloc[i].hour == 23:
            # Ensure the X window is in bounds
            if i + x_offsets[0] < 0 or i + x_offsets[1] >= n:
                continue

            # Define the reference for the Y window as the row after the marker (i+1)
            y_start_index = i + 1 + y_offsets[0]
            y_end_index = i + 1 + y_offsets[1]
            if y_end_index >= n:
                continue

            # Verify that the row immediately after the marker is 00:00 (start of day D+1)
            if times.iloc[i+1].hour != 0:
                continue

            # Also ensure that all rows in the Y window belong to the same day (day D+1)
            pred_day = times.iloc[i].date() + pd.Timedelta(days=1)
            valid_y = all(times.iloc[j].date() == pred_day for j in range(i+1, i+1+y_window_length))
            if not valid_y:
                continue

            # Extract windows using the relative offsets (adding 1 to y reference)
            X_window = x_scaled[i + x_offsets[0] : i + x_offsets[1] + 1]
            Y_window = y_scaled[y_start_index : y_end_index + 1]

            samples_X.append(X_window)
            samples_Y.append(Y_window)

    return np.array(samples_X), np.array(samples_Y)


def quantile_loss(q, name):
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q * e, (q - 1) * e))
    loss.__name__ = f'{name}_label'
    return loss


def load_trained_model(path, QUANTILES):

    quantiles_dict = {f'q{q:02}': q / 100 for q in QUANTILES}
    custom_objects = {f'{name}_label': quantile_loss(q, name) for name, q in quantiles_dict.items()}

    with custom_object_scope(custom_objects):
        model = load_model(path, custom_objects=custom_objects)

    return model


def split_and_scale_data(df, country, train_start, val_start, test_start, test_end, feature_cols=None):

    # Ensure the datetime column is in datetime format and set as index
    df = df.copy()
    df['CET'] = pd.to_datetime(df['CET'], utc=True).dt.tz_convert('CET')
    df = df.set_index('CET', drop=False)
    # Fill missing values using forward fill then backward fill
    df = df.ffill().bfill() 
    
    # Define the target column and feature columns for the country.
    # The target is e.g., 'LT-DA_price' and features are all columns containing 'LT' except the target.
    target_col = [col for col in df.columns if 'DA_price' in col and country in col][0] # extract the string instead of using list
    
    if feature_cols == None: # use all features
        feature_cols_selected = [col for col in df.columns if country in col and col != target_col]
        print('all available features are used: \n', feature_cols)
    else: 
        feature_cols_selected = [col for col in df.columns if any(keyword in col for keyword in feature_cols) and country in col]
        if 'DayOfWeek' in feature_cols:
            feature_cols_selected = feature_cols_selected + ['DayOfWeek']
        if 'Holiday' in feature_cols:
            feature_cols_selected = feature_cols_selected + ['Holiday']
    print(f'🗂️ [Feature] {feature_cols_selected}')

    # Split data by date ranges.
    train_start = convert_datetime(train_start)
    val_start = convert_datetime(val_start)
    test_start = convert_datetime(test_start)
    test_end = convert_datetime(test_end)
    
    train_df = df[(df['CET'] >= train_start) & (df['CET'] < val_start)]
    val_df = df[(df['CET'] >= val_start) & (df['CET'] < test_start)]
    test_df = df[(df['CET'] >= test_start) & (df['CET'] < test_end)]
    
    # Extract features (x) and target (y)
    x_train = train_df[feature_cols_selected]
    x_val   = val_df[feature_cols_selected]
    x_test  = test_df[feature_cols_selected]
    
    y_train = train_df[target_col]
    y_val   = val_df[target_col]
    y_test  = test_df[target_col]
    
    # Preserve the CET times for segmentation
    train_times = train_df['CET']
    val_times   = val_df['CET']
    test_times  = test_df['CET']
    
    # Scale features using RobustScaler (no need to keep the scaler)
    x_scaler = RobustScaler()

    if len(feature_cols) == 1:
        x_train_scaled = x_scaler.fit_transform(x_train.values.reshape(-1, 1))
        x_val_scaled   = x_scaler.transform(x_val.values.reshape(-1, 1))
        x_test_scaled  = x_scaler.transform(x_test.values.reshape(-1, 1))

    else:
        x_train_scaled = x_scaler.fit_transform(x_train)
        x_val_scaled   = x_scaler.transform(x_val)
        x_test_scaled  = x_scaler.transform(x_test)
    
    # Scale target using RobustScaler; record this scaler for inverse transformation later.
    y_scaler = RobustScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_val_scaled   = y_scaler.transform(y_val.values.reshape(-1, 1)).flatten()
    y_test_scaled  = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()
    
    return (x_train_scaled, x_val_scaled, x_test_scaled,
            y_train_scaled, y_val_scaled, y_test_scaled,
            train_times, val_times, test_times,
            y_scaler)



def create_daily_windows(x_scaled, y_scaled, times, x_offsets=(-72, -1), y_offsets=(0, 23)):

    samples_X = []
    samples_Y = []
    times = pd.to_datetime(times)
    n = len(times)

    # Calculate window lengths (inclusive of endpoints)
    y_window_length = y_offsets[1] - y_offsets[0] + 1

    # Iterate over indices looking for a marker (time == 23:00)
    for i in range(n):
        if times.iloc[i].hour == 23:
            # Ensure the X window is in bounds
            if i + x_offsets[0] < 0 or i + x_offsets[1] >= n:
                continue

            # Define the reference for the Y window as the row after the marker (i+1)
            y_start_index = i + 1 + y_offsets[0]
            y_end_index = i + 1 + y_offsets[1]
            if y_end_index >= n:
                continue

            # Verify that the row immediately after the marker is 00:00 (start of day D+1)
            if times.iloc[i+1].hour != 0:
                continue

            # Also ensure that all rows in the Y window belong to the same day (day D+1)
            pred_day = times.iloc[i].date() + pd.Timedelta(days=1)
            valid_y = all(times.iloc[j].date() == pred_day for j in range(i+1, i+1+y_window_length))
            if not valid_y:
                continue

            # Extract windows using the relative offsets (adding 1 to y reference)
            X_window = x_scaled[i + x_offsets[0] : i + x_offsets[1] + 1]
            Y_window = y_scaled[y_start_index : y_end_index + 1]

            samples_X.append(X_window)
            samples_Y.append(Y_window)

    return np.array(samples_X), np.array(samples_Y)

def align_samples(X_price, Y_price, X_res):
    n_price = len(X_price)
    n_res = len(X_res)

    if n_price > n_res:
        diff = n_price - n_res
        print(f"🔧 [Align] Dropping first {diff} price samples to match multivariate")
        X_price = X_price[diff:]
        Y_price = Y_price[diff:]
        return X_price, Y_price, X_res
    
    elif n_res > n_price:
        diff = n_res - n_price
        print(f"🔧 [Align] Dropping first {diff} multivariate samples to match price")
        X_res = X_res[diff:]
        return X_price, Y_price, X_res
    
    else:
        return X_price, Y_price, X_res
    

def prepare_univariate_and_multivariate_data(european_energy_df, region, train_start, val_start, test_start, test_end, input_features, prediction_horizon=23, look_back_window=-24):
    # Univariate (price only)
    (x_train_scaled_u, x_val_scaled_u, x_test_scaled_u,
     y_train_scaled_u, y_val_scaled_u, y_test_scaled_u,
     train_times_u, val_times_u, test_times_u,
     y_scaler) = split_and_scale_data(european_energy_df, region, train_start, val_start, test_start, test_end, ['DA_price'])

    X_train_price, Y_train_price = create_daily_windows(x_train_scaled_u, y_train_scaled_u, train_times_u, x_offsets=(-24, -1), y_offsets=(0, prediction_horizon))
    X_val_price, Y_val_price     = create_daily_windows(x_val_scaled_u, y_val_scaled_u, val_times_u, x_offsets=(-24, -1), y_offsets=(0, prediction_horizon))
    X_test_price, Y_test_price   = create_daily_windows(x_test_scaled_u, y_test_scaled_u, test_times_u, x_offsets=(-24, -1), y_offsets=(0, prediction_horizon))

    # Multivariate features
    (x_train_scaled_m, x_val_scaled_m, x_test_scaled_m,
     y_train_scaled_m, y_val_scaled_m, y_test_scaled_m,
     train_times_m, val_times_m, test_times_m,
     _) = split_and_scale_data(european_energy_df, region, train_start, val_start, test_start, test_end, input_features)

    X_train_res, _ = create_daily_windows(x_train_scaled_m, y_train_scaled_m, train_times_m, x_offsets=(look_back_window, 23), y_offsets=(0, prediction_horizon))
    X_val_res, _   = create_daily_windows(x_val_scaled_m, y_val_scaled_m, val_times_m, x_offsets=(look_back_window, 23), y_offsets=(0, prediction_horizon))
    X_test_res, _  = create_daily_windows(x_test_scaled_m, y_test_scaled_m, test_times_m, x_offsets=(look_back_window, 23), y_offsets=(0, prediction_horizon))

    # Align univariate and multivariate samples
    X_train_price, Y_train_price, X_train_res = align_samples(X_train_price, Y_train_price, X_train_res)
    X_val_price, Y_val_price, X_val_res       = align_samples(X_val_price, Y_val_price, X_val_res)
    X_test_price, Y_test_price, X_test_res    = align_samples(X_test_price, Y_test_price, X_test_res)

    return (X_train_price,  X_train_res,  Y_train_price,
            X_val_price,    X_val_res,    Y_val_price,
            X_test_price,   X_test_res,   Y_test_price, 
            y_scaler)


def generate_adjacency_matrix(filter_countries=None):
    country_codes = ['AT', 'BE', 'BG', 'CZ', 'DE_LU', 'DK_1', 'DK_2', 
                    'EE', 'ES', 'FI', 'FR', 'GR', 'HR', 'HU', 
                    'IT_1', 'IT_2', 'IT_3', 'IT_4', 'IT_5','IT_6','IT_7', 
                    'LT', 'LV', 'NL', 'NO_1', 'NO_2', 'NO_3', 'NO_4', 'NO_5',
                    'PL', 'PT', 'RO', 'SE_1', 'SE_2', 'SE_3', 'SE_4', 'SI', 'SK']
    
    adjacency_dict = {
        'AT': ['AT', 'CZ', 'DE_LU', 'HU', 'IT_4', 'SI'],
        'BE': ['BE', 'DE_LU', 'FR', 'NL'],
        'BG': ['BG', 'GR', 'RO'],
        'CZ': ['AT', 'CZ', 'DE_LU', 'PL', 'SK'],
        'DE_LU': ['AT', 'BE', 'CZ', 'DK_1', 'DK_2', 'DE_LU', 'FR', 'NL', 'NO_2', 'PL', 'SE_4'],
        'DK_1': ['DE_LU', 'DK_1', 'DK_2', 'NL', 'NO_2', 'SE_3'],
        'DK_2': ['DE_LU', 'DK_1', 'DK_2', 'SE_4'],
        'EE': ['EE', 'FI', 'LV'],
        'ES': ['ES', 'FR', 'PT'],
        'FI': ['EE', 'FI', 'NO_4', 'SE_1', 'SE_3'],
        'FR': ['BE', 'DE_LU', 'ES', 'FR', 'IT_4'],
        'GR': ['BG', 'GR', 'IT_7'],
        'HR': ['HR', 'HU', 'SI'],
        'HU': ['AT', 'HR', 'HU', 'RO', 'SI', 'SK'],
        'IT_1': ['IT_1', 'IT_6', 'IT_7'],
        'IT_2': ['IT_2', 'IT_3', 'IT_4'],
        'IT_3': ['IT_2', 'IT_3', 'IT_5', 'IT_7'],
        'IT_4': ['AT', 'FR', 'IT_2', 'IT_4', 'SI'],
        'IT_5': ['IT_3', 'IT_5'],
        'IT_6': ['IT_1', 'IT_6'],
        'IT_7': ['GR', 'IT_1', 'IT_3', 'IT_7'],
        'LT': ['LT', 'LV', 'PL', 'SE_4'],
        'LV': ['EE', 'LT', 'LV'],
        'NL': ['BE', 'DK_1', 'DE_LU', 'NL', 'NO_2'],
        'NO_1': ['NO_1', 'NO_2', 'NO_3', 'NO_5', 'SE_3'],
        'NO_2': ['DE_LU', 'DK_1', 'NL', 'NO_1', 'NO_2', 'NO_5'],
        'NO_3': ['NO_1', 'NO_3', 'NO_4', 'NO_5', 'SE_2'],
        'NO_4': ['FI', 'NO_3', 'NO_4', 'SE_1', 'SE_2'],
        'NO_5': ['NO_1', 'NO_2', 'NO_3', 'NO_5'],
        'PL': ['CZ', 'DE_LU', 'LT', 'PL', 'SE_4', 'SK'],
        'PT': ['ES', 'PT'],
        'RO': ['BG', 'HU', 'RO'],
        'SE_1': ['FI', 'NO_4', 'SE_1', 'SE_2'],
        'SE_2': ['NO_3', 'NO_4', 'SE_1', 'SE_2', 'SE_3'],
        'SE_3': ['DK_1', 'FI', 'NO_1', 'SE_2', 'SE_3', 'SE_4'],
        'SE_4': ['DE_LU', 'DK_2', 'LT', 'PL', 'SE_3', 'SE_4'],
        'SI': ['AT', 'HR', 'HU', 'IT_4', 'SI'],
        'SK': ['CZ', 'HU', 'PL', 'SK'],
    }

    # reconstructing the adjacency matrix from dictionary
    adj_df = pd.DataFrame(0, index=country_codes, columns=country_codes)
    for key, values in adjacency_dict.items():
        for value in values:
            adj_df.loc[key, value] = 1
    if filter_countries:
        adj_df = adj_df.loc[filter_countries, filter_countries]

    is_symmetric = (adj_df.values == adj_df.values.T).all()
    print(f"🗺️ Check if adj matrix symmetric: {is_symmetric}")

    #A = tf.constant(adj_df, dtype=tf.float32) # Return TensorFlow tensor
    A = adj_df.values.astype(np.float32)  # Return NumPy array
    return A


def build_region_branch(region, num_layer, hidden_dim, input_shape_price, input_shape_res):

    # Get input shapes
    input_price = Input(shape=input_shape_price, name=f'{region}_input_price')
    input_res   = Input(shape=input_shape_res, name=f'{region}_input_res')
    
    # Projection: Match the time dimensions 
    target_time_dim = max(input_shape_price[0], input_shape_res[0])
    # Permute so that the time dimension comes second.
    price_permuted = Permute((2, 1))(input_price)  # shape: (batch, num_features_price, num_timesteps_price)
    res_permuted   = Permute((2, 1))(input_res)      # shape: (batch, num_features_res, num_timesteps_res)
    
    price_projected = Dense(target_time_dim, activation='linear')(price_permuted)
    res_projected   = Dense(target_time_dim, activation='linear')(res_permuted)
    
    # Permute back to shape (batch, timesteps, features)
    representation_price = Permute((2, 1))(price_projected)
    representation_res   = Permute((2, 1))(res_projected)
    
    # Representation Learning
    for _ in range(num_layer):
        representation_price = Dense(hidden_dim, activation='swish')(representation_price)
        representation_res   = Dense(hidden_dim, activation='swish')(representation_res)

    # Combine and Flatten
    combined = Add()([representation_price, representation_res])
    combined = Flatten()(combined)
    
    return Model(
        inputs={f'{region}_input_price': input_price, f'{region}_input_res': input_res},
        outputs=combined,
        name=f'{region}_branch'
    )


def lr_schedule(epoch):
    initial_lr = 4e-3
    decay_factor = 0.95
    decay_interval = 10

    num_decays = epoch // decay_interval
    return initial_lr * (decay_factor ** num_decays)


def pinball_loss(y_true, y_pred, quantile):
    errors = y_true - y_pred
    loss = np.where(errors >= 0, quantile * errors, (quantile - 1) * errors)
    return np.mean(loss)


def compute_regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def compute_quantile_crossing_rate(y_pred_array):

    n_samples, output_dim, n_quantiles = y_pred_array.shape
    total_points = n_samples * output_dim
    violation_count = 0
    for i in range(n_samples):
        for t in range(output_dim):
            quantile_preds = y_pred_array[i, t, :]
            if not np.all(np.diff(quantile_preds) >= 0):
                violation_count += 1
    crossing_rate = violation_count / total_points
    return crossing_rate


def compute_quantile_losses(y_true, y_pred_list, quantiles):
    # Flatten arrays to compute global errors for each quantile head.
    y_true_flat = y_true.flatten()
    quantile_losses = []
    for q, y_pred in zip(quantiles, y_pred_list):
        y_pred_flat = y_pred.flatten()
        loss = pinball_loss(y_true_flat, y_pred_flat, q)
        quantile_losses.append(loss)
    avg_quantile_loss = float(np.mean(quantile_losses))
    return quantile_losses, avg_quantile_loss
    

@register_keras_serializable()
class StackLayer(tf.keras.layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super(StackLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs, **kwargs):
        return tf.stack(inputs, axis=self.axis)

    def get_config(self):
        config = super(StackLayer, self).get_config()
        config.update({"axis": self.axis})
        return config

@register_keras_serializable()
class OnesMaskLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(OnesMaskLayer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return tf.ones_like(inputs[:, :, :1], dtype=tf.float32)

    def get_config(self):
        return super(OnesMaskLayer, self).get_config()


@register_keras_serializable()
class AdjacencyMatMulLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AdjacencyMatMulLayer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        A, X = inputs  # A: (batch_size, num_nodes, num_nodes), X: (batch_size, num_nodes, feature_dim)
        return tf.linalg.matmul(A, X)

    def get_config(self):
        config = super(AdjacencyMatMulLayer, self).get_config()
        return config


@register_keras_serializable()
def quantile_loss(q, name=None):
    def loss_fn(y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q * e, (q - 1) * e))
    loss_fn.__name__ = name if name else f"quantile_{int(q*100)}"
    return loss_fn


def load_trained_model(path, QUANTILES):

    quantiles_dict = {f'q{q:02}': q / 100 for q in QUANTILES}

    quantiles_dict = {f'q{q:02}': q / 100 for q in QUANTILES}
    custom_objects = {
        "StackLayer": StackLayer,
        "OnesMaskLayer": OnesMaskLayer,
    }

    # register quantile loss
    for name, q in quantiles_dict.items():
        label = f'{name}_label'
        custom_objects[label] = quantile_loss(q, label)
    
    with custom_object_scope(custom_objects):
        model = load_model(path, custom_objects=custom_objects, compile=False)
        
    return model


def form_region_dic(regions, european_energy_df, 
                    train_start, val_start, test_start, test_end,
                    input_features, prediction_horizon, look_back_window):
    
    region_data = {}
    for region in regions:
        print(f'🔄 [{region}] Processing starts ...')
        (X_train_price,  X_train_res,  Y_train_price,
        X_val_price,    X_val_res,    Y_val_price,
        X_test_price,   X_test_res,   Y_test_price, 
        y_scaler) = prepare_univariate_and_multivariate_data(
            european_energy_df, region,
            train_start, val_start, test_start, test_end,
            input_features, prediction_horizon, look_back_window)

        region_data[region] = {
            'X_train_price': X_train_price,
            'Y_train_price': Y_train_price,
            'X_train_res': X_train_res,
            'X_val_price': X_val_price,
            'Y_val_price': Y_val_price,
            'X_val_res': X_val_res,
            'X_test_price': X_test_price,
            'Y_test_price': Y_test_price,
            'X_test_res': X_test_res,
            'y_scaler': y_scaler,
        }
        print(f'🎉 Processing finished. \n')
    return region_data


def HierarchicalQuantileHead_perRegion(
    shared_rep,      # (batch, hidden_dim)
    quantiles,       # e.g. [10,50,90]
    output_dim,      # e.g. 24
    prefix           # e.g. "AT"
):
    # Sort & find median
    sorted_q   = sorted(quantiles)
    median_idx = sorted_q.index(50)

    # median
    out_median = Dense(output_dim, name=f"{prefix}_q50_label")(shared_rep)
    outputs = {50: out_median}

    # upper quantiles with smooth Softplus steps
    prev = out_median
    for q in sorted_q[median_idx+1:]:
        pre  = Dense(output_dim, name=f"{prefix}_q{q:02}_pre_project")(shared_rep)
        step = AbsActivation(name=f"{prefix}_q{q:02}_step")(pre)
        
        o    = Add(name=f"{prefix}_q{q:02}_label")([prev, step])
        outputs[q] = o
        prev = o

    # lower quantiles with smooth Softplus steps
    prev = out_median
    for q in reversed(sorted_q[:median_idx]):
        pre  = Dense(output_dim, name=f"{prefix}_q{q:02}_pre_project")(shared_rep)
        step = AbsActivation(name=f"{prefix}_q{q:02}_step")(pre)
        o    = Subtract(name=f"{prefix}_q{q:02}_label")([prev, step])
        outputs[q] = o
        prev = o

    # return in original order
    return [outputs[q] for q in quantiles]


@register_keras_serializable()
class QuantileStack(Layer):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        # `inputs` is a Python list of tensors; we stack them along `self.axis`
        return tf.stack(inputs, axis=self.axis)

    def get_config(self):
        # ensure `axis` is saved in config so that this layer is serializable
        config = super().get_config()
        config.update({"axis": self.axis})
        return config
    

@register_keras_serializable()
def stack_quantile_loss(quantiles):
    def loss(y_true, y_pred):
        total_loss = 0.0
        K = float(len(quantiles))
        for idx, q in enumerate(quantiles):
            e  = y_true[..., idx] - y_pred[..., idx]
            qf = q / 100.0
            total_loss += tf.reduce_mean(tf.maximum(qf * e, (qf - 1.0) * e))
        return total_loss / K
    return loss


def run_model(model, QUANTILES, output_regions, train_inputs, val_inputs, y_train_dict, y_val_dict, epoch, batch_size, model_path, show_progress_bar=True):

    # build losses
    losses = {
        f"{region}_quantiles_stack": stack_quantile_loss(QUANTILES)
        for region in output_regions
    }

    model.compile(optimizer=Adam(learning_rate=1e-3), loss=losses)

    checkpoint_callback = ModelCheckpoint(
        model_path,
        monitor='val_loss',
        save_freq="epoch",
        save_best_only=True,
        mode='min',
        verbose=show_progress_bar
    )
    lr_scheduler_callback = LearningRateScheduler(lr_schedule)
    print(f"🧠 #Params: {model.count_params()}")

    history = model.fit(
        x=train_inputs,
        y=y_train_dict,
        validation_data=(val_inputs, y_val_dict),
        callbacks=[checkpoint_callback, lr_scheduler_callback],
        epochs=epoch,
        batch_size=batch_size,
        verbose=False
    )
    return model, history, model.count_params()


@register_keras_serializable()
class GatherMaskRows(Layer):

    def __init__(self, indices, **kwargs):
        super().__init__(**kwargs)

        # We convert to a tf.constant so it’s baked into the graph
        self.indices = tf.constant(indices, dtype=tf.int32)

    def call(self, x):

        # x has shape (batch, n_nodes, hidden_dim)
        # We pick only the rows in `self.indices` along axis=1.
        return tf.gather(x, self.indices, axis=1)

    def get_config(self):

        # Include `indices` so that serialization works
        config = super().get_config()
        config.update({"indices": self.indices.numpy().tolist()})
        return config


'''
----------------
Test performance 
----------------
'''


def test_performance_multi_head(best_model, test_inputs, Y_test_price, region_data, quantiles, regions):

    n_regions = len(regions)
    n_samples, _, output_dim = Y_test_price.shape
    n_quantiles = len(quantiles)

    # Inverse-transform the true Y_test_price for each region, y_test_original has shape (n_samples, n_regions, output_dim).
    y_test_original_list = []
    for r in range(n_regions):
        y_r = Y_test_price[:, r, :]  # shape = (n_samples, output_dim)
        y_r_orig = region_data[regions[r]]['y_scaler'].inverse_transform(y_r)
        y_test_original_list.append(y_r_orig)
    y_test_original = np.stack(y_test_original_list, axis=1)

    print("🚀 Testing performance per region...")
    start_time = time.time()

    # Predict with the model. This returns a list of length n_regions. Each element has shape (n_samples, output_dim, n_quantiles).
    y_pred_list = best_model.predict(test_inputs) 
    print(f'Prediction shape: {np.shape(y_pred_list)}')
    
    end_time = time.time()
    inference_time = end_time - start_time

    arr = np.array(y_pred_list)
    # If we have one country, predict() returns shape (n_samples, output_dim, n_quantiles) → ndim == 3
    # If we have multiple countries, predict() returns shape (n_regions, n_samples, output_dim, n_quantiles) → ndim == 4
    if arr.ndim == 3:
        # single‐region case → wrap into a list of length 1
        y_pred_list = [arr]
    elif arr.ndim == 4 and arr.shape[0] == n_regions:
        # multi‐region case → split along axis=0
        y_pred_list = [arr[r] for r in range(n_regions)]

    # Inverse-transform each predicted quantile slice for each region. For region r, `pred_rescaled[r]` has shape (n_samples, output_dim, n_quantiles).
    pred_rescaled = []
    for r in range(n_regions):
        y_pred_scaled_region = y_pred_list[r] # shape = (n_samples, output_dim, n_quantiles)
        region_pred_rescaled_slices = []
        for q_idx in range(n_quantiles):
            # slice out (n_samples, output_dim)
            slice_scaled = y_pred_scaled_region[..., q_idx]
            slice_rescaled = region_data[regions[r]]['y_scaler'].inverse_transform(slice_scaled)
            region_pred_rescaled_slices.append(slice_rescaled)

        # Stack back into shape (n_samples, output_dim, n_quantiles)
        region_pred_rescaled = np.stack(region_pred_rescaled_slices, axis=-1)
        pred_rescaled.append(region_pred_rescaled)

    # Now compute metrics per region
    results = {}
    for r, region_name in enumerate(regions):
        region_metrics = {}
        # Ground truth for this region, already inverse-transformed:
        y_true_region = y_test_original[:, r, :]           # shape = (n_samples, output_dim)
        # Predicted for this region:
        y_pred_region = pred_rescaled[r]                   # shape = (n_samples, output_dim, n_quantiles)

        # Pinball loss (average quantile loss) for each quantile
        quant_losses = []
        for q_idx, q in enumerate(quantiles):
            # Flatten both arrays to (n_samples * output_dim,)
            true_flat = y_true_region.flatten()
            pred_flat = y_pred_region[..., q_idx].flatten()
            e = true_flat - pred_flat
            loss_vals = np.maximum(q * e, (q - 1.0) * e)
            quant_losses.append(np.mean(loss_vals))
        region_metrics['quantile_losses'] = quant_losses
        region_metrics['avg_quantile_loss'] = float(np.mean(quant_losses))

        # RMSE, MAE, R² on the median quantile (q = 0.5)
        if 0.5 in quantiles:
            median_idx = quantiles.index(0.5)
        else:
            median_idx = 0
        pred_median = y_pred_region[..., median_idx].flatten()
        true_flat   = y_true_region.flatten()

        # RMSE
        mse = np.mean((true_flat - pred_median)**2)
        rmse = float(np.sqrt(mse))
        # MAE
        mae = float(np.mean(np.abs(true_flat - pred_median)))
        # R²
        ss_res = np.sum((true_flat - pred_median)**2)
        ss_tot = np.sum((true_flat - np.mean(true_flat))**2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        region_metrics['rmse'] = rmse
        region_metrics['mae']  = mae
        region_metrics['r2']   = r2

        # Quantile‐crossing rate
        # Stack all quantile slices to shape (n_samples, output_dim, n_quantiles)
        # and count how often a lower quantile > a higher quantile at the same (sample, horizon).
        stacked_q = y_pred_region  # shape = (n_samples, output_dim, n_quantiles)

        crossing = np.zeros((n_samples, output_dim), dtype=bool)
        for i in range(n_quantiles - 1):
            lower = stacked_q[..., i]
            upper = stacked_q[..., i + 1]
            crossing |= (lower > upper)

        # crossing is a boolean array of shape (n_samples, output_dim)
        crossing_rate = float(np.mean(crossing))
        region_metrics['quantile_crossing_rate'] = crossing_rate

        results[region_name] = region_metrics

    results['inference_time'] = inference_time

    # Print
    avg_aql_total = 0
    region_count = 0

    print(f"Inference time: {inference_time:.2f} seconds\n")
    for region_name in regions:
        rm = results[region_name]
        print(f"Region {region_name}:")
        print(f"  Avg quantile loss:       {rm['avg_quantile_loss']:.4f}")
        print(f"  Quantile losses (q={quantiles}): {', '.join(f'{l:.4f}' for l in rm['quantile_losses'])}")
        print(f"  RMSE (q=0.5):            {rm['rmse']:.4f}")
        print(f"  MAE  (q=0.5):            {rm['mae']:.4f}")
        print(f"  R²   (q=0.5):            {rm['r2']:.4f}")
        print(f"  Quantile crossing rate:  {rm['quantile_crossing_rate']*100:.2f}%\n")
        
        # Accumulate AQL for averaging
        avg_aql_total += rm['avg_quantile_loss']
        region_count += 1

    # Print overall average AQL
    if region_count > 0:
        avg_aql = avg_aql_total / region_count
        print(f"Average AQL across all regions: {avg_aql:.4f}\n")

    return results


'''
----------------------------------------------
Adj matrix and degree rows
----------------------------------------------
'''


def get_degree_rows(adj_df, target_country, degree):
    # K -breadth search to find all countries within a certain degree of separation
    countries = adj_df.index.tolist()
    idx_map = {country: i for i, country in enumerate(countries)}
    
    visited = set()
    queue = deque([(target_country, 0)])

    while queue:
        node, d = queue.popleft()
        if node in visited or d > degree:
            continue
        visited.add(node)
        neighbors = adj_df.loc[node]
        for neighbor in neighbors[neighbors == 1].index:
            queue.append((neighbor, d + 1))

    # Build binary output matrix
    binary_matrix = np.zeros_like(adj_df.values, dtype=np.float32)
    for country in visited:
        row_idx = idx_map[country]
        binary_matrix[row_idx, :] = 1  # entire row is 1

    return binary_matrix


def get_valid_masking(input_regions, target_country, DEFAULT_DEGREES):
    degree = DEFAULT_DEGREES[target_country]
    adj_matrix = generate_adjacency_matrix(filter_countries=input_regions)

    adj_df = pd.DataFrame(adj_matrix, index=input_regions, columns=input_regions)
    binary_result = get_degree_rows(adj_df, target_country=target_country, degree=degree)

    non_zero_rows = np.where(binary_result.sum(axis=1) > 0)[0]
    non_zero_country_names = [input_regions[i] for i in non_zero_rows]

    print(f"✅ Target country: {target_country}, Degree = {degree}, Neighbors: {non_zero_country_names}")
    return binary_result


def save_as_csv(results, hyper_setup, history, quantiles, pre_path):

    train_start, val_start, test_start, look_back_window, prediction_horizon, model_params, seed, select_mode = hyper_setup 

    rows = []

    # Extract inference time if it's stored globally (not per-region)
    global_inference_time = results.get("inference_time", None)

    for region, metrics in results.items():
        if not isinstance(metrics, dict):
            continue  # skip global inference_time or other non-dict entries

        row = {"region": region}

        # Unpack each quantile loss into its own column
        q_losses = metrics.get("quantile_losses", [])
        for q_test, loss_test in zip(quantiles, q_losses):
            row[f"quantile_loss_{q_test}"] = loss_test

        # Copy scalar metrics
        for key in ("avg_quantile_loss", "quantile_crossing_rate", "rmse", "mae", "r2"):
            row[key] = metrics.get(key, None)

        # If per-region inference time exists, use it. Else fallback to global.
        row["inference_time"] = metrics.get("inference_time", global_inference_time)

        row["look_back_window"] = look_back_window
        row["prediction_horizon"] = prediction_horizon
        row["model_params"] = model_params
        row["seed"] = seed
        row['select_mode'] = select_mode
        rows.append(row)

    df = pd.DataFrame(rows)

    # Add last row as average
    avg_row = df.drop(columns=['region']).mean(numeric_only=True)
    avg_row['region'] = 'Avg.'
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    path = pre_path + f"Result/{train_start}_{val_start}_{test_start}_lbw{look_back_window}_ph{prediction_horizon}_seed{seed}_{select_mode}.csv"
    df.to_csv(path, index=False)


'''
----------------------------------------------
PriceFM 
----------------------------------------------
'''

def build_node_features(region_data, input_regions, num_layer, hidden_dim):
    region_inputs = {}
    region_outputs = {}
    for region in input_regions:
        shp_price = region_data[region]['X_train_price'].shape[1:]
        shp_res   = region_data[region]['X_train_res'].shape[1:]
        inp_price = Input(shape=shp_price, name=f'{region}_input_price')
        inp_res   = Input(shape=shp_res,   name=f'{region}_input_res')
        region_inputs[region] = {
            f'{region}_input_price': inp_price,
            f'{region}_input_res':   inp_res
        }
        branch = build_region_branch(region, num_layer, hidden_dim, shp_price, shp_res)
        region_outputs[region] = branch(region_inputs[region])

    node_features = StackLayer(
        axis=1,
        name="stack_input_regions"
    )([region_outputs[r] for r in input_regions])

    return node_features, region_inputs


def get_decay_mask(input_regions, target_country, curvature):

    # Build adjacency DataFrame
    adj_matrix = generate_adjacency_matrix(filter_countries=input_regions)
    adj_df = pd.DataFrame(adj_matrix, index=input_regions, columns=input_regions)

    # Map regions to indices
    region_to_idx = {region: i for i, region in enumerate(input_regions)}
    decay_vector = np.zeros(len(input_regions), dtype=np.float32)
    region_degree = {}

    # BFS to compute graph distances
    queue = deque([(target_country, 0)])
    visited = set()
    max_seen = 0

    while queue:
        node, degree = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        max_seen = max(max_seen, degree)

        if node in region_to_idx:
            region_degree[node] = degree

        # Enqueue neighbors
        for neighbor, connected in adj_df.loc[node].items():
            if connected and neighbor not in visited:
                queue.append((neighbor, degree + 1))

    # Determine normalization cap based on observed distances
    D = max_seen if max_seen > 0 else 1

    # Compute normalized decay weights
    for region, degree in region_degree.items():
        idx = region_to_idx[region]
        if curvature > 0:
            lam = 1- curvature
            # Convex: fast early decay
            if lam == 1.0:
                w = 1.0
            else:
                w = (lam**degree - lam**D) / (1 - lam**D)
        elif curvature < 0:
            lam = 1-(-curvature)
            # Concave: flat early, sudden drop late
            if lam == 1.0:
                w = 1.0
            else:
                w = (1 - lam**(D - degree)) / (1 - lam**D)
        else:
            # Linear decay
            w = 1 - degree / D

        decay_vector[idx] = float(w)

    # Diagnostic output
    print(f"✅ Target: {target_country}, curvature={curvature}, max_distance={D}")
    for region, degree in region_degree.items():
        idx = region_to_idx[region]
        print(f" - {region}: degree={degree}, weight={decay_vector[idx]:.4f}")

    return decay_vector


def get_decay_factor_for_region(region):
    # optimized curvature mapping per region based on val loss
    DECAY_FACTORS = {
        'AT': 0.8,
        'BE': 0.8,
        'BG': 1.0,
        'CZ': 0.8,
        'DE_LU': 0.0,
        'DK_1': 0.2,
        'DK_2': 0.4,
        'EE': -0.6,
        'ES': 1.0,
        'FI': 1.0,
        'FR': 1.0,
        'GR': 1.0,
        'HR': -0.2,
        'HU': 0.0,
        'IT_1': 1.0,
        'IT_2': 1.0,
        'IT_3': 1.0,
        'IT_4': 1.0,
        'IT_5': 1.0,
        'IT_6': 0.8,
        'IT_7': 1.0,
        'LT': 0.2,
        'LV': 0.4,
        'NL': 0.8,
        'NO_1': 0.8,
        'NO_2': 0.8,
        'NO_3': 0.8,
        'NO_4': 0.6,
        'NO_5': 0.8,
        'PL': 1.0,
        'PT': 1.0,
        'RO': 1.0,
        'SE_1': 0.2,
        'SE_2': 0.8,
        'SE_3': 1.0,
        'SE_4': 1.0,
        'SI': 0.2,
        'SK': 0.0,
    }

    try:
        return DECAY_FACTORS[region]
    except KeyError:
        raise ValueError(f"No decay factor defined for region '{region}'")


def aggregate_regions(node_features, input_regions, output_regions):
    flattened = {}

    for region in output_regions:

        # inject prior graph knowledge to produce decay mask
        decay_factor = get_decay_factor_for_region(region)
        decay_mask = get_decay_mask(input_regions, region, decay_factor)
        decay_mask = tf.constant(decay_mask, dtype=node_features.dtype)[None, :, None]
        masked = node_features * decay_mask

        # produce weighted average feature representation
        avg_weighted = GlobalAveragePooling1D()(masked) # could mutiply the #rows
        avg_weights  = GlobalAveragePooling1D()(decay_mask) # could mutiply the #rows
        normalized = avg_weighted  / avg_weights # the #rows is cancelled after division
        
        # flatten the pooled output
        flattened[region] = Flatten(name=f'flatten_{region}')(normalized)
        print(np.shape(flattened[region]))
    return flattened


def build_data(region_data, input_regions, output_regions, QUANTILES):

    # per-region train/val inputs
    train_inputs = {}
    val_inputs   = {}
    for region in input_regions:
        train_inputs[f"{region}_input_price"] = region_data[region]["X_train_price"]
        train_inputs[f"{region}_input_res"]   = region_data[region]["X_train_res"]
        val_inputs  [f"{region}_input_price"] = region_data[region]["X_val_price"]
        val_inputs  [f"{region}_input_res"]   = region_data[region]["X_val_res"]

    # build the y‐dicts for quantiles
    y_train_dict = {}
    y_val_dict   = {}
    n_q = len(QUANTILES)
    for region in output_regions:
        Ytr = region_data[region]["Y_train_price"]
        Yv  = region_data[region]["Y_val_price"]
        Ytr_rep = np.repeat(Ytr[..., None], n_q, axis=-1)
        Yv_rep  = np.repeat(Yv[..., None],  n_q, axis=-1)
        name = f"{region}_quantiles_stack"
        y_train_dict[name] = Ytr_rep.astype(np.float32)
        y_val_dict  [name] = Yv_rep.astype(np.float32)

    # convert to lists in output_regions order
    y_train_list = [
        y_train_dict[f"{region}_quantiles_stack"]
        for region in output_regions
    ]
    y_val_list   = [
        y_val_dict  [f"{region}_quantiles_stack"]
        for region in output_regions
    ]

    return train_inputs, val_inputs, y_train_list, y_val_list


def MultiHeadQuantileHead_perRegion(
    shared_rep,      # e.g. projected[region], shape = (batch, hidden_dim)
    quantiles,       # e.g. [10, 50, 90]
    output_dim,      # e.g. 24
    prefix           # e.g. "RegionA"
    ):

    outputs = []
    for q in quantiles:
        head = Dense(
            output_dim,
            name=f"{prefix}_q{q:02}_label"
        )(shared_rep)
        outputs.append(head)
    return outputs


def build_model(region_data, input_regions, output_regions, num_layer, hidden_dim, QUANTILES):

    # build node features
    node_features, region_inputs = build_node_features(
        region_data, input_regions, num_layer, hidden_dim
    )

    # aggregate per-output-region
    flattened = aggregate_regions(node_features, input_regions, output_regions)

    # collect inputs
    all_inputs = {}
    for r in input_regions:
        all_inputs.update(region_inputs[r])

    # collect outputs / build quantile heads
    all_outputs = []
    output_dim = region_data[input_regions[0]]['Y_train_price'].shape[1]
    for region in output_regions:
        head_list = HierarchicalQuantileHead_perRegion(flattened[region], QUANTILES, output_dim, region)
        stacked_q = QuantileStack(axis=-1, name=f"{region}_quantiles_stack")(head_list)
        all_outputs.append(stacked_q)

    return Model(inputs=list(all_inputs.values()), outputs=all_outputs)


def evaluate_model_unseen(region_data, input_regions, output_regions, model_path, QUANTILES):

    # Prepare test_inputs for price & residual for each input_region
    test_inputs = {}
    for region in input_regions:
        test_inputs[f"{region}_input_price"] = region_data[region]["X_test_price"]
        test_inputs[f"{region}_input_res"]   = region_data[region]["X_test_res"]

    # Stack true test prices across output_regions → shape = (n_test, n_regions, output_dim)
    y_test_stacked = np.stack(
        [region_data[r]["Y_test_price"] for r in output_regions],
        axis=1
    )

    # Load the trained model
    foundation_model = load_trained_model(model_path, QUANTILES)

    # Call testing function
    results = test_performance_multi_head(
        foundation_model,
        test_inputs,
        y_test_stacked,
        region_data,
        [q / 100.0 for q in QUANTILES],
        output_regions
    )
    return results


def load_region_data(pre_path, train_start, val_start, test_start, look_back_window, prediction_horizon):

    filename = f"{train_start}_{val_start}_{test_start}_lbw{look_back_window}_ph{prediction_horizon}.pkl"
    filepath = os.path.join(pre_path, "Data", filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Pickle file not found: {filepath}")

    with open(filepath, "rb") as f:
        region_data = pickle.load(f)

    print(f"🚀 Loaded region_data!")
    return region_data


'''
----------------------------------------------
ablation study
----------------------------------------------
'''


# Concatenation
def ablation_study_build_region_branch_concat(region, num_layer, hidden_dim, input_shape_price, input_shape_res):

    # Get input shapes
    input_price = Input(shape=input_shape_price, name=f'{region}_input_price')
    input_res   = Input(shape=input_shape_res, name=f'{region}_input_res')
    
    representation_price = Flatten()(input_price)
    representation_res   = Flatten()(input_res)

    # Representation Learning
    for _ in range(num_layer):
        representation_price = Dense(hidden_dim, activation='swish')(representation_price)
        representation_res   = Dense(hidden_dim, activation='swish')(representation_res)

    combined = Concatenate()([representation_price, representation_res])
    
    return Model(
        inputs={f'{region}_input_price': input_price, f'{region}_input_res': input_res},
        outputs=combined,
        name=f'{region}_branch'
    )


def ablation_study_build_node_features_concat(region_data, input_regions, num_layer, hidden_dim):
    region_inputs = {}
    region_outputs = {}
    for region in input_regions:
        shp_price = region_data[region]['X_train_price'].shape[1:]
        shp_res   = region_data[region]['X_train_res'].shape[1:]
        inp_price = Input(shape=shp_price, name=f'{region}_input_price')
        inp_res   = Input(shape=shp_res,   name=f'{region}_input_res')
        region_inputs[region] = {
            f'{region}_input_price': inp_price,
            f'{region}_input_res':   inp_res
        }
        branch = ablation_study_build_region_branch_concat(region, num_layer, hidden_dim, shp_price, shp_res)
        region_outputs[region] = branch(region_inputs[region])

    node_features = StackLayer(
        axis=1,
        name="stack_input_regions"
    )([region_outputs[r] for r in input_regions])

    return node_features, region_inputs


def build_model_ablation_study_concat_feature(region_data, input_regions, output_regions, num_layer, hidden_dim, QUANTILES):

    # build node features
    node_features, region_inputs = ablation_study_build_node_features_concat(
        region_data, input_regions, num_layer, hidden_dim
    )

    # aggregate per-output-region
    flattened = aggregate_regions(node_features, input_regions, output_regions)

    # collect inputs
    all_inputs = {}
    for r in input_regions:
        all_inputs.update(region_inputs[r])

    # collect outputs / build quantile heads
    all_outputs = []
    output_dim = region_data[input_regions[0]]['Y_train_price'].shape[1]
    for region in output_regions:
        head_list = HierarchicalQuantileHead_perRegion(flattened[region], QUANTILES, output_dim, region)
        stacked_q = QuantileStack(axis=-1, name=f"{region}_quantiles_stack")(head_list)
        all_outputs.append(stacked_q)

    return Model(inputs=list(all_inputs.values()), outputs=all_outputs)


#cross-attention
def ablation_study_build_region_branch_cross_attention(
    region,
    num_layer,
    hidden_dim,
    input_shape_price,
    input_shape_res,
    num_heads=4
):
    # Get input shapes
    input_price = Input(shape=input_shape_price, name=f'{region}_input_price')
    input_res   = Input(shape=input_shape_res, name=f'{region}_input_res')
    
    # Projection: Match the time dimensions 
    target_time_dim = max(input_shape_price[0], input_shape_res[0])

    # Permute so that the time dimension comes second.
    price_permuted = Permute((2, 1))(input_price)  # shape: (batch, num_features_price, num_timesteps_price)
    res_permuted   = Permute((2, 1))(input_res)      # shape: (batch, num_features_res, num_timesteps_res)
    
    price_projected = Dense(target_time_dim, activation='linear')(price_permuted)
    res_projected   = Dense(target_time_dim, activation='linear')(res_permuted)
    
    # Permute back to shape (batch, timesteps, features)
    repr_price = Permute((2, 1))(price_projected)
    repr_res   = Permute((2, 1))(res_projected)
    
    for _ in range(num_layer):
        repr_price = Dense(hidden_dim, activation='swish')(repr_price)
        repr_res   = Dense(hidden_dim, activation='swish')(repr_res)

    # Cross‑Attention: price attends to res
    attn_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=hidden_dim // num_heads,
        name=f'{region}_cross_attn'
    )(query=repr_price, value=repr_res, key=repr_res)
    
    # optional residual connection + layer norm
    combined = Add(name=f'{region}_cross_attn_residual')([repr_price, attn_output])
    combined = LayerNormalization(name=f'{region}_cross_attn_norm')(combined)
    combined = Flatten()(combined)

    return Model(
        inputs={f'{region}_input_price': input_price, f'{region}_input_res': input_res},
        outputs=combined,
        name=f'{region}_branch'
    )


def ablation_study_build_node_features_cross_attention(region_data, input_regions, num_layer, hidden_dim):
    region_inputs = {}
    region_outputs = {}
    for region in input_regions:
        shp_price = region_data[region]['X_train_price'].shape[1:]
        shp_res   = region_data[region]['X_train_res'].shape[1:]
        inp_price = Input(shape=shp_price, name=f'{region}_input_price')
        inp_res   = Input(shape=shp_res,   name=f'{region}_input_res')
        region_inputs[region] = {
            f'{region}_input_price': inp_price,
            f'{region}_input_res':   inp_res
        }
        branch = ablation_study_build_region_branch_cross_attention(region, num_layer, hidden_dim, shp_price, shp_res)
        region_outputs[region] = branch(region_inputs[region])

    node_features = StackLayer(
        axis=1,
        name="stack_input_regions"
    )([region_outputs[r] for r in input_regions])

    return node_features, region_inputs


def build_model_ablation_study_cross_attention(region_data, input_regions, output_regions, num_layer, hidden_dim, QUANTILES):

    # build node features
    node_features, region_inputs = ablation_study_build_node_features_cross_attention(
        region_data, input_regions, num_layer, hidden_dim
    )

    # aggregate per-output-region
    flattened = aggregate_regions(node_features, input_regions, output_regions)

    # collect inputs
    all_inputs = {}
    for r in input_regions:
        all_inputs.update(region_inputs[r])

    # collect outputs / build quantile heads
    all_outputs = []
    output_dim = region_data[input_regions[0]]['Y_train_price'].shape[1]
    for region in output_regions:
        head_list = HierarchicalQuantileHead_perRegion(flattened[region], QUANTILES, output_dim, region)
        stacked_q = QuantileStack(axis=-1, name=f"{region}_quantiles_stack")(head_list)
        all_outputs.append(stacked_q)

    return Model(inputs=list(all_inputs.values()), outputs=all_outputs)


def ablation_study_random_decay_mask(input_regions):

    # Generate a random decay mask with one value per input region drawn uniformly from [0, 1].
    decay_vector = np.random.uniform(low=0.0, high=1.0, size=len(input_regions)).astype(np.float32)

    # Diagnostic output
    print("🎲 Random decay weights:")
    for region, weight in zip(input_regions, decay_vector):
        print(f" - {region}: {weight:.4f}")

    return decay_vector


def ablation_study_randomize_graph_decay_aggregate_regions(node_features, input_regions, output_regions):
    flattened = {}

    for region in output_regions:

        # inject prior graph knowledge to produce decay mask
        #decay_factor = ablation_study_randomize_graph_degree_get_decay_factor_for_region(region)
        decay_mask = ablation_study_random_decay_mask(input_regions)
        decay_mask = tf.constant(decay_mask, dtype=node_features.dtype)[None, :, None]
        masked = node_features * decay_mask

        # produce weighted average feature representation
        avg_weighted = GlobalAveragePooling1D()(masked) # could mutiply the #rows
        avg_weights  = GlobalAveragePooling1D()(decay_mask) # could mutiply the #rows
        normalized = avg_weighted  / avg_weights # the #rows is cancelled after division
        # flatten the pooled output
        flattened[region] = Flatten(name=f'flatten_{region}')(normalized)
        print(np.shape(flattened[region]))
    return flattened


def build_model_ablation_study_randomize_graph_decay(region_data, input_regions, output_regions, num_layer, hidden_dim, QUANTILES):

    # build node features
    node_features, region_inputs = build_node_features(
        region_data, input_regions, num_layer, hidden_dim
    )

    # aggregate per-output-region
    flattened = ablation_study_randomize_graph_decay_aggregate_regions(node_features, input_regions, output_regions)

    # collect inputs
    all_inputs = {}
    for r in input_regions:
        all_inputs.update(region_inputs[r])

    # collect outputs / build quantile heads
    all_outputs = []
    output_dim = region_data[input_regions[0]]['Y_train_price'].shape[1]
    for region in output_regions:
        head_list = HierarchicalQuantileHead_perRegion(flattened[region], QUANTILES, output_dim, region)
        stacked_q = QuantileStack(axis=-1, name=f"{region}_quantiles_stack")(head_list)
        all_outputs.append(stacked_q)

    return Model(inputs=list(all_inputs.values()), outputs=all_outputs)


def ablation_study_remove_graph_decay_aggregate_regions(node_features, input_regions, output_regions):
    
    flattened = {}
    for region in output_regions:

        # produce weighted average feature representation
        aggregated = GlobalAveragePooling1D()(node_features) 

        # flatten the pooled output
        flattened[region] = Flatten(name=f'flatten_{region}')(aggregated)

    return flattened


def build_model_ablation_study_remove_graph_decay(region_data, input_regions, output_regions, num_layer, hidden_dim, QUANTILES):

    # build node features
    node_features, region_inputs = build_node_features(
        region_data, input_regions, num_layer, hidden_dim
    )

    # aggregate per-output-region
    flattened = ablation_study_remove_graph_decay_aggregate_regions(node_features, input_regions, output_regions)

    # collect inputs
    all_inputs = {}
    for r in input_regions:
        all_inputs.update(region_inputs[r])

    # collect outputs / build quantile heads
    all_outputs = []
    output_dim = region_data[input_regions[0]]['Y_train_price'].shape[1]
    for region in output_regions:
        head_list = HierarchicalQuantileHead_perRegion(flattened[region], QUANTILES, output_dim, region)
        stacked_q = QuantileStack(axis=-1, name=f"{region}_quantiles_stack")(head_list)
        all_outputs.append(stacked_q)

    return Model(inputs=list(all_inputs.values()), outputs=all_outputs)


def normal_multi_head(
    shared_rep,      # e.g. projected[region], shape = (batch, hidden_dim)
    quantiles,       # e.g. [10, 50, 90]
    output_dim,      # e.g. 24
    prefix           # e.g. "AT"
):
    # One Dense layer per quantile
    outputs = []
    for q in quantiles:
        o = Dense(output_dim, name=f"{prefix}_q{q:02}_label")(shared_rep)
        outputs.append(o)

    return outputs


def build_model_ablation_study_normal_multi_head(region_data, input_regions, output_regions, num_layer, hidden_dim, QUANTILES):

    # build node features
    node_features, region_inputs = build_node_features(
        region_data, input_regions, num_layer, hidden_dim
    )

    # aggregate per-output-region
    flattened = aggregate_regions(node_features, input_regions, output_regions)

    # collect inputs
    all_inputs = {}
    for r in input_regions:
        all_inputs.update(region_inputs[r])

    # collect outputs / build quantile heads
    all_outputs = []
    output_dim = region_data[input_regions[0]]['Y_train_price'].shape[1]
    for region in output_regions:
        head_list = normal_multi_head(flattened[region], QUANTILES, output_dim, region)
        stacked_q = QuantileStack(axis=-1, name=f"{region}_quantiles_stack")(head_list)
        all_outputs.append(stacked_q)

    return Model(inputs=list(all_inputs.values()), outputs=all_outputs)


def ablation_study_HQH_relu(
    shared_rep,      # (batch, hidden_dim)
    quantiles,       # e.g. [10,50,90]
    output_dim,      # e.g. 24
    prefix           # e.g. "AT"
):
    # Sort & find median
    sorted_q   = sorted(quantiles)
    median_idx = sorted_q.index(50)

    # median
    out_median = Dense(output_dim, name=f"{prefix}_q50_label")(shared_rep)
    outputs = {50: out_median}

    # upper quantiles with smooth Softplus steps
    prev = out_median
    for q in sorted_q[median_idx+1:]:
        pre  = Dense(output_dim, name=f"{prefix}_q{q:02}_pre_project")(shared_rep)
        step = Activation('relu', name=f"{prefix}_q{q:02}_step")(pre)
        o    = Add(name=f"{prefix}_q{q:02}_label")([prev, step])
        outputs[q] = o
        prev = o

    # lower quantiles with smooth Softplus steps
    prev = out_median
    for q in reversed(sorted_q[:median_idx]):
        pre  = Dense(output_dim, name=f"{prefix}_q{q:02}_pre_project")(shared_rep)
        step = Activation('relu', name=f"{prefix}_q{q:02}_step")(pre)
        o    = Subtract(name=f"{prefix}_q{q:02}_label")([prev, step])
        outputs[q] = o
        prev = o

    # return in original order
    return [outputs[q] for q in quantiles]


def build_model_ablation_study_HQH_relu(region_data, input_regions, output_regions, num_layer, hidden_dim, QUANTILES):

    # build node features
    node_features, region_inputs = build_node_features(
        region_data, input_regions, num_layer, hidden_dim
    )

    # aggregate per-output-region
    flattened = aggregate_regions(node_features, input_regions, output_regions)

    # collect inputs
    all_inputs = {}
    for r in input_regions:
        all_inputs.update(region_inputs[r])

    # collect outputs / build quantile heads
    all_outputs = []
    output_dim = region_data[input_regions[0]]['Y_train_price'].shape[1]
    for region in output_regions:
        head_list = ablation_study_HQH_relu(flattened[region], QUANTILES, output_dim, region)
        stacked_q = QuantileStack(axis=-1, name=f"{region}_quantiles_stack")(head_list)
        all_outputs.append(stacked_q)

    return Model(inputs=list(all_inputs.values()), outputs=all_outputs)


@register_keras_serializable()
class AbsActivation(Layer):
    def __init__(self, **kwargs):
        super(AbsActivation, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.abs(inputs)

    def get_config(self):
        return super(AbsActivation, self).get_config()


def ablation_study_HQH_abs(
    shared_rep,      # (batch, hidden_dim)
    quantiles,       # e.g. [10,50,90]
    output_dim,      # e.g. 24
    prefix           # e.g. "AT"
):
    # Sort & find median
    sorted_q   = sorted(quantiles)
    median_idx = sorted_q.index(50)

    # median
    out_median = Dense(output_dim, name=f"{prefix}_q50_label")(shared_rep)
    outputs = {50: out_median}

    # upper quantiles with smooth Softplus steps
    prev = out_median
    for q in sorted_q[median_idx+1:]:
        pre  = Dense(output_dim, name=f"{prefix}_q{q:02}_pre_project")(shared_rep)
        step = AbsActivation(name=f"{prefix}_q{q:02}_step")(pre)
        
        o    = Add(name=f"{prefix}_q{q:02}_label")([prev, step])
        outputs[q] = o
        prev = o

    # lower quantiles with smooth Softplus steps
    prev = out_median
    for q in reversed(sorted_q[:median_idx]):
        pre  = Dense(output_dim, name=f"{prefix}_q{q:02}_pre_project")(shared_rep)
        step = AbsActivation(name=f"{prefix}_q{q:02}_step")(pre)
        o    = Subtract(name=f"{prefix}_q{q:02}_label")([prev, step])
        outputs[q] = o
        prev = o

    # return in original order
    return [outputs[q] for q in quantiles]


def build_model_ablation_study_HQH_abs(region_data, input_regions, output_regions, num_layer, hidden_dim, QUANTILES):

    # build node features
    node_features, region_inputs = build_node_features(
        region_data, input_regions, num_layer, hidden_dim
    )

    # aggregate per-output-region
    flattened = aggregate_regions(node_features, input_regions, output_regions)

    # collect inputs
    all_inputs = {}
    for r in input_regions:
        all_inputs.update(region_inputs[r])

    # collect outputs / build quantile heads
    all_outputs = []
    output_dim = region_data[input_regions[0]]['Y_train_price'].shape[1]
    for region in output_regions:
        head_list = ablation_study_HQH_abs(flattened[region], QUANTILES, output_dim, region)
        stacked_q = QuantileStack(axis=-1, name=f"{region}_quantiles_stack")(head_list)
        all_outputs.append(stacked_q)

    return Model(inputs=list(all_inputs.values()), outputs=all_outputs)


@register_keras_serializable()
class SquareActivation(Layer):
    def __init__(self, **kwargs):
        super(SquareActivation, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.square(inputs)

    def get_config(self):
        return super(SquareActivation, self).get_config()
    

def ablation_study_HQH_square(
    shared_rep,      # (batch, hidden_dim)
    quantiles,       # e.g. [10,50,90]
    output_dim,      # e.g. 24
    prefix           # e.g. "AT"
):
    # Sort & find median
    sorted_q   = sorted(quantiles)
    median_idx = sorted_q.index(50)

    # median
    out_median = Dense(output_dim, name=f"{prefix}_q50_label")(shared_rep)
    outputs = {50: out_median}

    # upper quantiles with smooth Softplus steps
    prev = out_median
    for q in sorted_q[median_idx+1:]:
        pre  = Dense(output_dim, name=f"{prefix}_q{q:02}_pre_project")(shared_rep)
        step = SquareActivation(name=f"{prefix}_q{q:02}_step")(pre)
        
        o    = Add(name=f"{prefix}_q{q:02}_label")([prev, step])
        outputs[q] = o
        prev = o

    # lower quantiles with smooth Softplus steps
    prev = out_median
    for q in reversed(sorted_q[:median_idx]):
        pre  = Dense(output_dim, name=f"{prefix}_q{q:02}_pre_project")(shared_rep)
        step = SquareActivation(name=f"{prefix}_q{q:02}_step")(pre)
        o    = Subtract(name=f"{prefix}_q{q:02}_label")([prev, step])
        outputs[q] = o
        prev = o

    # 4) return in original order
    return [outputs[q] for q in quantiles]


def build_model_ablation_study_HQH_square(region_data, input_regions, output_regions, num_layer, hidden_dim, QUANTILES):

    # build node features
    node_features, region_inputs = build_node_features(
        region_data, input_regions, num_layer, hidden_dim
    )

    # aggregate per-output-region
    flattened = aggregate_regions(node_features, input_regions, output_regions)

    # collect inputs
    all_inputs = {}
    for r in input_regions:
        all_inputs.update(region_inputs[r])

    # collect outputs / build quantile heads
    all_outputs = []
    output_dim = region_data[input_regions[0]]['Y_train_price'].shape[1]
    for region in output_regions:
        head_list = ablation_study_HQH_square(flattened[region], QUANTILES, output_dim, region)
        stacked_q = QuantileStack(axis=-1, name=f"{region}_quantiles_stack")(head_list)
        all_outputs.append(stacked_q)

    return Model(inputs=list(all_inputs.values()), outputs=all_outputs)


def build_model_mode(select_mode, region_data, input_regions, output_regions, num_layer, hidden_dim, QUANTILES):
    
    # our optimized model structure
    if select_mode == 'optimal':
        model = build_model(region_data, input_regions, output_regions, num_layer, hidden_dim, QUANTILES)
    
    elif select_mode == 'ablation_study_concat_feature':
        model = build_model_ablation_study_concat_feature(region_data, input_regions, output_regions, num_layer, hidden_dim, QUANTILES)
    
    elif select_mode == 'ablation_study_cross_attention':
        model = build_model_ablation_study_cross_attention(region_data, input_regions, output_regions, num_layer, hidden_dim, QUANTILES)
    
    elif select_mode == 'ablation_study_randomize_graph_decay':
        model = build_model_ablation_study_randomize_graph_decay(region_data, input_regions, output_regions, num_layer, hidden_dim, QUANTILES)
    
    elif select_mode == 'ablation_study_remove_graph_decay':
        model = build_model_ablation_study_remove_graph_decay(region_data, input_regions, output_regions, num_layer, hidden_dim, QUANTILES)
    
    elif select_mode == 'ablation_study_normal_multi_head':
        model = build_model_ablation_study_normal_multi_head(region_data, input_regions, output_regions, num_layer, hidden_dim, QUANTILES)
    
    elif select_mode == 'ablation_study_HQH_relu':
        model = build_model_ablation_study_HQH_relu(region_data, input_regions, output_regions, num_layer, hidden_dim, QUANTILES)
    
    elif select_mode == 'ablation_study_HQH_abs':
        model = build_model_ablation_study_HQH_abs(region_data, input_regions, output_regions, num_layer, hidden_dim, QUANTILES)
    
    elif select_mode == 'ablation_study_HQH_square':
        model = build_model_ablation_study_HQH_square(region_data, input_regions, output_regions, num_layer, hidden_dim, QUANTILES)
    
    return model


# generate intermediate data
def generate_region_data_pickles(
    data_splits,
    look_back_windows,
    prediction_horizons,
    regions,
    european_energy_df,
    input_features,
    pre_path,
):

    for data_split in data_splits:
        train_start, val_start, test_start, test_end = data_split

        for look_back_window in look_back_windows:
            for prediction_horizon in prediction_horizons:
                print(f"Processing data split: {data_split}, look_back_window: {look_back_window}, prediction_horizon: {prediction_horizon}")

                # Generate region data
                region_data = form_region_dic(
                    regions,
                    european_energy_df,
                    train_start,
                    val_start,
                    test_start,
                    test_end,
                    input_features,
                    prediction_horizon,
                    look_back_window
                )

                # Construct filename and filepath
                filename = f"{train_start}_{val_start}_{test_start}_lbw{look_back_window}_ph{prediction_horizon}.pkl"
                filepath = os.path.join(pre_path, "Data", filename)

                # Ensure directory exists
                os.makedirs(os.path.dirname(filepath), exist_ok=True)

                # Save to pickle
                with open(filepath, "wb") as f:
                    pickle.dump(region_data, f)

                print(f"Saved: {filepath}")


# below are the functions that will be used in the multiprocessing setup
def run_single_config(args):
    (pre_path, train_start, val_start, test_start,
     look_back_window, prediction_horizon,
     input_regions, output_regions,
     QUANTILES, model_path, epoch, batch_size, show_progress_bar, seed, 
     select_mode, num_layer, hidden_dim) = args

    region_data = load_region_data(pre_path, train_start, val_start, test_start, look_back_window, prediction_horizon)
    train_inputs, val_inputs, y_train_dict, y_val_dict = build_data(region_data, input_regions, output_regions, QUANTILES)
    model = build_model_mode(select_mode, region_data, input_regions, output_regions, num_layer, hidden_dim, QUANTILES)
    best_model, history, model_params = run_model(model, QUANTILES, output_regions, train_inputs, val_inputs, y_train_dict, y_val_dict, epoch, batch_size, model_path, show_progress_bar)
    results = evaluate_model_unseen(region_data, input_regions, output_regions, model_path, QUANTILES)
    hyper_setup = (train_start, val_start, test_start, look_back_window, prediction_horizon, model_params, seed, select_mode)
    save_as_csv(results, hyper_setup, history, QUANTILES, pre_path)

    del model, best_model, history, model_params, results, region_data
    K.clear_session()
    gc.collect()


def run_all(data_splits, look_back_windows, prediction_horizons, input_regions, TARGET_REGIONS, seeds, select_modes, pre_path, QUANTILES, model_path, epoch, batch_size, show_progress_bar, num_layer, hidden_dim):

    for data_split in data_splits:
        train_start, val_start, test_start, _ = data_split
        for look_back_window in look_back_windows:
            for prediction_horizon in prediction_horizons:
                for output_regions in TARGET_REGIONS:
                    for seed in seeds:
                        for select_mode in select_modes:
                            print(f"Running configuration: {train_start}, {val_start}, {test_start}, "
                                  f"look_back_window={look_back_window}, prediction_horizon={prediction_horizon}, "
                                  f"output_regions={output_regions}, seed={seed}, select_mode={select_mode}")
                            
                            args = (
                                pre_path, train_start, val_start, test_start,
                                look_back_window, prediction_horizon,
                                input_regions, output_regions, QUANTILES, 
                                model_path, epoch, batch_size, show_progress_bar, seed, select_mode, num_layer, hidden_dim
                            )
                            p = Process(target=run_single_config, args=(args,))
                            p.start()
                            p.join()
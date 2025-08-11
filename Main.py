from PriceFM import *

pre_path = train_status('local') # 'local' or 'cloud', cloud for GoogleColab
model_path = pre_path+"Model/PriceFM.keras" # location to save the model
european_energy_df = pre_path+'Data/EU_Spatiotemporal_Energy_Data.csv' # location of dataset
european_energy_df = pd.read_csv(european_energy_df)

# master lists of input features [not change]
input_features = ['day_ahead_Solar', 'day_ahead_Wind Onshore', 'day_ahead_Wind Offshore', 'Forecasted Load']

# master lists of region codes [not change]
all_regions = ['AT',    'BE',    'BG',   'CZ',    'DE_LU',   'DK_1',  'DK_2',                 
               'EE',    'ES',    'FI',   'FR',    'GR',      'HR',    'HU',                  
               'IT_1',  'IT_2',  'IT_3', 'IT_4',  'IT_5',    'IT_6',  'IT_7', 
               'LT',    'LV',    'NL',   'NO_1',  'NO_2',    'NO_3',  'NO_4',   'NO_5', 
               'PL',    'PT',    'RO',   'SE_1',  'SE_2',    'SE_3',  'SE_4',   'SI',   'SK']

# quantiles for prediction, e.g., 10%, 50%, and 90% quantiles [can change]
QUANTILES = [10, 50, 90] 

# input regions of the model [can change]
input_regions =  all_regions 

# output regions for prediction;
# here can be a list of lists, e.g., [['AT'], ['AT', 'BE']] for individual region or combinations
# if specified as [['AT'], ['AT', 'BE']], then we train 2 models, one to predict AT, and another to predict AT and BE. [can change]
TARGET_REGIONS = [all_regions] 

# optimal model hyperparameters [can change]
num_layer, hidden_dim, epoch, batch_size, show_progress_bar = (3, 24, 20, 8, True)

# data splits for training, validation, and testing [can change]
data_splits = [('2022-01-01', '2024-01-01', '2024-07-01', '2025-01-01')]

# time windows for input and output [can change]
look_back_windows = [-24] # the last 24 hours of data as input. [-24, -23, ...-1]
prediction_horizons = [23] # the next 24 hours to predict / look-forward window size for load, solar, and wind. [0, 1, ...23]

# seeds for randomness [can change]
seeds = [42] 

# run the optimal model or ablation study [can change]
select_modes = ['optimal']

# main function to run all steps
if __name__ == "__main__":

    # when first time excecute the main, run the following function "generate_region_data_pickles"
    # to generate intermediate data to avoid repeated processing; otherwise, comment it out.
    generate_region_data_pickles(data_splits, look_back_windows, prediction_horizons, 
                                all_regions, european_energy_df, input_features, pre_path)
    
    # run the main function: load processed intermediate data, train, validate, and test the model
    run_all(data_splits, look_back_windows, prediction_horizons, 
            input_regions, TARGET_REGIONS, seeds, select_modes, 
            pre_path, QUANTILES, model_path, epoch, batch_size, show_progress_bar, num_layer, hidden_dim)



import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

TYPES = ['effect_na', 'min_ves', 'ves_na_kru', 'moment']

# Define feature engineering function
def add_features(df):
    """Add trigonometric features and polar coordinates."""
    df["sin_Incl."] = np.sin(df["Incl"])
    df["cos_Incl."] = np.cos(df["Incl"])
    df["tan_Incl."] = np.tan(df["Incl"])
    df["sin_Azim."] = np.sin(df["Azim"])
    df["cos_Azim."] = np.cos(df["Azim"])
    df["tan_Azim."] = np.tan(df["Azim"])
    df["Local_polar_angle"] = np.arctan2(df["Local_N_Coord"], df["Local_E_Coord"])
    df["Local_polar_r"] = np.sqrt(df["Local_N_Coord"]**2 + df["Local_E_Coord"]**2)
    df["Global_polar_angle"] = np.arctan2(df["Global_N_Coord"], df["Global_E_Coord"])
    df["Global_polar_r"] = np.sqrt(df["Global_N_Coord"]**2 + df["Global_E_Coord"]**2)
    return df

# Define the base directory
BASE_DIR = Path(__file__).resolve(strict=True).parent

# Load the model and scalers using joblib
def load_models_and_scalers(model_type):
    """Load the Keras neural network model and XGBoost model along with their scalers."""
    nn_model = tf.keras.models.load_model(f'{BASE_DIR}/nn_model_{model_type}.h5')
    xgb_model = joblib.load(f'{BASE_DIR}/xgb_model_{model_type}.pkl')
    x_scaler = joblib.load(f'{BASE_DIR}/x_scaler_{model_type}.pkl')
    y_scaler = joblib.load(f'{BASE_DIR}/y_scaler_{model_type}.pkl')
    return nn_model, xgb_model, x_scaler, y_scaler

# Define output columns for each type
def get_output_columns(model_type):
    """Get output columns based on the model type."""
    if model_type == 'effect_na':
        return ['Грузоподъёмность вышки(effect_na)', 'Бурение ротором(effect_na)', 'Спиральный изгиб(без вращения)(effect_na)',
                'Подъём(effect_na)', 'Синусоидальный изгиб(все операции)(effect_na)', 'Спуск(effect_na)', 
                'Бурение ГЗД(effect_na)', 'Спиральный изгиб(с вращением)(effect_na)', 'Предел натяжения(effect_na)'] 
    elif model_type == 'ves_na_kru':
        return ['Грузоподъёмность вышки(ves_na_kru)', 'Бурение ротором(ves_na_kru)', 'Подъём(ves_na_kru)', 
                'Спуск(ves_na_kru)', 'Бурение ГЗД(ves_na_kru)', 'Мин. вес до спирального изгиба (спуск)(ves_na_kru)',
                'Макс. вес до предела текучести (подъём)(ves_na_kru)']  
    elif model_type == 'moment':
        return ['Бурение ротором(moment)', 'Подъём(moment)', 'Make-up Torque(moment)', 'Спуск(moment)', 'Момент свинчивания(moment)']  
    elif model_type == 'min_ves':
        return ['Мин. вес на долоте до спирального изгиба (бурение ротором)(min_ves)', 'Мин. вес на долоте до синусоидального изгиба (бурение ГЗД)(min_ves)',
                'Мин. вес на долоте до синусоидального изгиба (бурение ротором)(min_ves)', 'Мин. вес на долоте до спирального изгиба (бурение ГЗД)(min_ves)']  
    
    return ['prediction']

# Make prediction with the model
def make_prediction(input_data):
    """Make predictions using the loaded models and scalers."""
    prediction_dfs = []
    
    # Apply feature engineering
    input_data = add_features(input_data)
    
    # Load models and scalers for each type
    for model_type in TYPES:
        nn_model, xgb_model, x_scaler, y_scaler = load_models_and_scalers(model_type)
             
        # Scale input data
        input_scaled = x_scaler.transform(input_data)

        # Make predictions using the neural network
        nn_predictions = nn_model.predict(input_scaled)

        # Combine predictions with original input dataset for XGBoost
        xgb_input = np.concatenate((input_scaled, nn_predictions), axis=1)

        # Make predictions using XGBoost
        xgb_predictions = xgb_model.predict(xgb_input)

        # Inverse scale the predictions
        prediction = y_scaler.inverse_transform(xgb_predictions)

        # Get unique output columns for the current type
        output_columns = get_output_columns(model_type)

        # Create a DataFrame for the predictions with appropriate columns
        predictions = pd.DataFrame(prediction, columns=output_columns)

        # Convert to long format
        predictions_long = predictions.melt(var_name='Variable', value_name='Predictions')

        # Add the current type to the dataframe
        prediction_dfs.append(predictions_long)
    
    # Concatenate all predictions dataframes
    concatenated_prediction_df = pd.concat(prediction_dfs, ignore_index=True)
    return concatenated_prediction_df

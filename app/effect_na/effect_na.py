import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler


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
def load_models_and_scalers():
    """Load the Keras neural network model and XGBoost model along with their scalers."""
    nn_model = tf.keras.models.load_model(f'{BASE_DIR}/nn_model_effect_na.h5')
    xgb_model = joblib.load(f'{BASE_DIR}/xgb_model_effect_na.pkl')
    x_scaler = joblib.load(f'{BASE_DIR}/x_scaler_effect_na.pkl')
    y_scaler = joblib.load(f'{BASE_DIR}/y_scaler_effect_na.pkl')
    return nn_model, xgb_model, x_scaler, y_scaler

# Define output columns for each type
def get_output_columns():
    """Get output columns based on the model type."""
    return ['Грузоподъёмность вышки', 'Бурение ротором', 'Спиральный изгиб(без вращения)',
                'Подъём', 'Синусоидальный изгиб(все операции)', 'Спуск', 
                'Бурение ГЗД', 'Спиральный изгиб(с вращением)', 'Предел натяжения'] 


# Make prediction with the model
def make_prediction_effect_na(input_data):
    """Make predictions using the loaded models and scalers."""
    # Ensure input_data is a DataFrame
    if isinstance(input_data, pd.DataFrame):
        # Apply feature engineering to the entire DataFrame
        input_data = add_features(input_data)
        
        # Load models and scalers for each type
        nn_model, xgb_model, x_scaler, y_scaler = load_models_and_scalers()
        
        # Scale input data
        input_scaled = x_scaler.transform(input_data)

        # Prepare an empty list to store predictions
        predictions = []
        
        # Loop through each row to make individual predictions
        for index, row in input_data.iterrows():
            # Reshape the row for the model
            row_scaled = input_scaled[index].reshape(1, -1)

            # Make predictions using the neural network
            nn_prediction = nn_model.predict(row_scaled)

            # Combine predictions for XGBoost
            xgb_input = np.concatenate((row_scaled, nn_prediction), axis=1)

            # Make predictions using XGBoost
            xgb_prediction = xgb_model.predict(xgb_input)

            # Inverse scale the predictions
            inverse_prediction = y_scaler.inverse_transform(xgb_prediction)

            # Store the prediction
            predictions.append(inverse_prediction[0])  # Append the prediction result
        
        # Get unique output columns for the current type
        output_columns = get_output_columns()

        # Create a DataFrame for the predictions
        predictions_df = pd.DataFrame(predictions, columns=output_columns)

        # Convert to long format
        predictions_long = predictions_df.melt(var_name='Variable', value_name='Predictions')
        
        return predictions_long
    else:
        raise ValueError("Input data must be a pandas DataFrame.")


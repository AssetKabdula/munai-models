from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
from collections import defaultdict
from app.effect_na.effect_na import make_prediction_effect_na
from app.ves_na_kru.ves_na_kru import make_prediction_ves_na_kru
from app.moment.moment import make_prediction_moment
from app.min_ves.min_ves import make_prediction_min_ves

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_methods=["*"],   # Allows all methods
    allow_headers=["*"],   # Allows all headers
)

# Define a request model that allows multiple values (as lists) for each feature
class CommonRequest(BaseModel):
    MD: List[float]
    Incl: List[float]
    Azim: List[float]
    Sub_Sea: List[float]
    TVD: List[float]
    Local_N_Coord: List[float]
    Local_E_Coord: List[float]
    Global_N_Coord: List[float]
    Global_E_Coord: List[float]
    Dogleg: List[float]
    Vertical_Section: List[float]
    Body_OD: List[float]
    Body_ID: List[float]
    Body_AvgJointLength: List[float]
    Stabilizer_Length: List[float]
    Stabilizer_OD: List[float]
    Stabilizer_ID: List[float]
    Weight: List[float]
    Coefficient_of_Friction: List[float]
    Minimum_Yield_Strength: List[float]

    class Config:
        schema_extra = {
            "example": {
                "MD": [1000, 1100],
                "Incl": [3, 4],
                "Azim": [4, 5],
                "Sub_Sea": [5, 6],
                "TVD": [43, 44],
                "Local_N_Coord": [3, 4],
                "Local_E_Coord": [4, 5],
                "Global_N_Coord": [4, 5],
                "Global_E_Coord": [5, 6],
                "Dogleg": [5, 6],
                "Vertical_Section": [5, 6],
                "Body_OD": [5, 6],
                "Body_ID": [5, 6],
                "Body_AvgJointLength": [5, 6],
                "Stabilizer_Length": [5, 6],
                "Stabilizer_OD": [5, 6],
                "Stabilizer_ID": [5, 6],
                "Weight": [5, 6],
                "Coefficient_of_Friction": [5, 6],
                "Minimum_Yield_Strength": [5, 6]
            }
        }

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Munai-Research prediction API"}

# Separate prediction endpoint for each model

@app.post("/predict/effect_na/")
def predict_effect_na(request: CommonRequest):
    # Convert the input data into a DataFrame for batch prediction
    input_data = create_dataframe(request)
    
    # Use the effect_na model to make a prediction
    prediction_df = make_prediction_effect_na(input_data)
    return organize_predictions(prediction_df)

@app.post("/predict/ves_na_kru/")
def predict_ves_na_kru(request: CommonRequest):
    # Convert the input data into a DataFrame for batch prediction
    input_data = create_dataframe(request)

    # Use the ves_na_kru model to make a prediction
    prediction_df = make_prediction_ves_na_kru(input_data)
    return organize_predictions(prediction_df)

@app.post("/predict/moment/")
def predict_moment(request: CommonRequest):
    # Convert the input data into a DataFrame for batch prediction
    input_data = create_dataframe(request)

    # Use the moment model to make a prediction
    prediction_df = make_prediction_moment(input_data)
    return organize_predictions(prediction_df)

@app.post("/predict/min_ves/")
def predict_min_ves(request: CommonRequest):
    # Convert the input data into a DataFrame for batch prediction
    input_data = create_dataframe(request)

    # Use the min_ves model to make a prediction
    prediction_df = make_prediction_min_ves(input_data)
    return organize_predictions(prediction_df)

# Helper functions

def create_dataframe(request: CommonRequest) -> pd.DataFrame:
    # Convert the input data into a DataFrame for batch prediction
    input_data = pd.DataFrame({
        'MD': request.MD,
        'Incl': request.Incl,
        'Azim': request.Azim,
        'Sub_Sea': request.Sub_Sea,
        'TVD': request.TVD,
        'Local_N_Coord': request.Local_N_Coord,
        'Local_E_Coord': request.Local_E_Coord,
        'Global_N_Coord': request.Global_N_Coord,
        'Global_E_Coord': request.Global_E_Coord,
        'Dogleg': request.Dogleg,
        'Vertical_Section': request.Vertical_Section,
        'Body_OD': request.Body_OD,
        'Body_ID': request.Body_ID,
        'Body_AvgJointLength': request.Body_AvgJointLength,
        'Stabilizer_Length': request.Stabilizer_Length,
        'Stabilizer_OD': request.Stabilizer_OD,
        'Stabilizer_ID': request.Stabilizer_ID,
        'Weight': request.Weight,
        'Coefficient_of_Friction': request.Coefficient_of_Friction,
        'Minimum_Yield_Stress': request.Minimum_Yield_Strength
    })

    # Check if the input data is valid and consistent
    if not all(len(lst) == len(request.MD) for lst in input_data.values.T):
        raise HTTPException(status_code=400, detail="All input lists must have the same length.")
    
    return input_data

def organize_predictions(prediction_df: pd.DataFrame) -> dict:
    # Group predictions by variable
    predictions_by_variable = defaultdict(list)

    # Loop through the predictions and append them for each variable
    for variable, prediction in zip(prediction_df["Variable"], prediction_df["Predictions"]):
        predictions_by_variable[variable].append(prediction)

    # Convert defaultdict back to a regular dictionary before returning
    return dict(predictions_by_variable)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
from collections import defaultdict
from app.model.model import make_prediction

# Initialize FastAPI app
app = FastAPI()

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

# Prediction endpoint for batch processing
@app.post("/predict/")
def predict(request: CommonRequest):

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

    # Use model to make prediction
    prediction_df = make_prediction(input_data)
    # Group predictions by variable
    predictions_by_variable = defaultdict(list)

    # Loop through the predictions and append them for each variable
    for variable, prediction in zip(prediction_df["Variable"], prediction_df["Predictions"]):
        predictions_by_variable[variable].append(prediction)

    # Convert defaultdict back to a regular dictionary before returning
    return dict(predictions_by_variable)
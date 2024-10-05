from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
from typing import Literal
import pandas as pd
import numpy as np
from app.model.model import make_prediction

# Initialize FastAPI app
app = FastAPI()

# Define a single request model for all types
class CommonRequest(BaseModel):
    MD: float
    Incl: float 
    Azim: float 
    Sub_Sea: float 
    TVD: float 
    Local_N_Coord: float 
    Local_E_Coord: float 
    Global_N_Coord: float 
    Global_E_Coord: float 
    Dogleg: float
    Vertical_Section: float 
    Body_OD: float 
    Body_ID: float 
    Body_AvgJointLength: float 
    Stabilizer_Length: float  
    Stabilizer_OD: float 
    Stabilizer_ID: float 
    Weight: float 
    Coefficient_of_Friction: float  
    Minimum_Yield_Stress: float  

    class Config:
        schema_extra = {
            "example": {
                "MD": 1000,
                "Incl": 3,
                "Azim": 4,
                "Sub_Sea": 5,
                "TVD": 43,
                "Local_N_Coord": 3,
                "Local_E_Coord": 4,
                "Global_N_Coord": 4,
                "Global_E_Coord": 5,
                "Dogleg": 5,
                "Vertical_Section": 5,
                "Body_OD": 5,
                "Body_ID": 5,
                "Body_AvgJointLength": 5,
                "Stabilizer_Length": 5,
                "Stabilizer_OD": 5,
                "Stabilizer_ID": 5,
                "Weight": 5,
                "Coefficient_of_Friction": 5,
                "Minimum_Yield_Stress": 5,
            }
        }

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Munai-Research prediction API"}

# Prediction endpoint
@app.post("/predict/")
def predict(request: CommonRequest):

    # Convert validated data to DataFrame (excluding 'current_type')
    input_data = pd.DataFrame([request.dict()])

    # Use model to make prediction
    try:
        prediction_df = make_prediction(input_data)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Return the prediction
    return dict(zip(prediction_df["Variable"], prediction_df["Predictions"]))

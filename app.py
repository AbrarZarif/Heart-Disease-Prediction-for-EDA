from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd   # <- added

# Initialize FastAPI app
app = FastAPI(title="Heart Disease Prediction API")

# Load the trained pipeline
with open("reg.pkl", "rb") as f:
    model = pickle.load(f)

# Define input schema
class PatientData(BaseModel):
    Cholesterol: float
    BP: float
    Max_HR: float
    Age: float
    Sex: int
    Chest_pain_type: int
    FBS_over_120: int
    EKG_results: int
    Exercise_angina: int
    ST_depression: float
    Slope_of_ST: int
    Num_vessels_fluro: int
    Thallium: int

#@app.post("/predict")
def predict(data: PatientData):
    # Convert input to dict
    data_dict = data.dict()
    
    # Mapping column names to match training
    data_dict["Max HR"] = data_dict.pop("Max_HR")
    data_dict["Chest pain type"] = data_dict.pop("Chest_pain_type")
    data_dict["Exercise angina"] = data_dict.pop("Exercise_angina")
    data_dict["Number of vessels fluro"] = data_dict.pop("Num_vessels_fluro")
    data_dict["FBS over 120"] = data_dict.pop("FBS_over_120")
    data_dict["EKG results"] = data_dict.pop("EKG_results")
    data_dict["Slope of ST"] = data_dict.pop("Slope_of_ST")
    data_dict["ST depression"] = data_dict.pop("ST_depression")
    
    df = pd.DataFrame([data_dict])
    
    # Engineer ischemia_risk
    def compute_ischemia_risk(row):
        if row["ST depression"] >= 2.0 and row["Slope of ST"] == 3:
            return "high"
        elif row["ST depression"] >= 1.0:
            return "moderate"
        else:
            return "low"

    df["ischemia_risk"] = df.apply(compute_ischemia_risk, axis=1)
    
    # One-hot encode ischemia_risk to match training columns
    df["ischemia_risk_low"] = (df["ischemia_risk"]=="low").astype(int)
    df["ischemia_risk_moderate"] = (df["ischemia_risk"]=="moderate").astype(int)
    df["ischemia_risk_high"] = (df["ischemia_risk"]=="high").astype(int)
    df = df.drop("ischemia_risk", axis=1)
    

    expected_cols = model.feature_names_in_  # Get expected feature names from the model
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0  # Add missing columns with default 0

    
    df = df[expected_cols]
    
    # Predict
    prediction_class = model.predict(df)[0]
    prediction_prob = model.predict_proba(df)[0][1]
    
    prediction = "Heart Disease Detected" if prediction_class==1 else "No Heart Disease Detected"
    
    return {
        "prediction": prediction,
        "probability": float(prediction_prob)
    }

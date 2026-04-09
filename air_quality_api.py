from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import joblib
import numpy as np
import os

#app setup
app = FastAPI(
    title="Air Quality Classification API",
    description=(
        "Classifies air quality using a Random Forest model. "
        "Input pollutant concentrations (pm25, pm10, no2, o3, co) in µg/m³ and get an AQI category. "
        "Model trained on balanced dataset with all 6 classes (Good to Hazardous)."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#class labels
CLASS_LABELS = {
    0: "Good",
    1: "Moderate",
    2: "Unhealthy for Sensitive Groups",
    3: "Unhealthy",
    4: "Very Unhealthy",
    5: "Hazardous",
}

CLASS_COLOR = {
    0: "#00e400",   # green
    1: "#ffff00",   # yellow
    2: "#ff7e00",   # orange
    3: "#ff0000",   # red
    4: "#8f3f97",   # purple
    5: "#7e0023",   # maroon
}

CLASS_ADVICE = {
    0: "Air quality is satisfactory. No restrictions for outdoor activities.",
    1: "Air quality is acceptable. Sensitive individuals should be cautious.",
    2: "Members of sensitive groups may experience health effects. General public is less likely to be affected.",
    3: "Everyone may begin to experience health effects. Limit prolonged outdoor exertion.",
    4: "Health alert: everyone may experience serious health effects. Avoid outdoor activities.",
    5: "Health emergency. Everyone is likely to be affected. Stay indoors.",
}


MINMAX_RANGES = {
    "pm25": (0.0, 500.0),
    "pm10": (0.0, 700.0),
    "no2":  (0.0, 2000.0),
    "o3":   (0.0, 500.0),
    "co":   (0.0, 50000.0),
}

def normalize_input(pm25: float, pm10: float, no2: float,
                    o3: float, co: float) -> np.ndarray:
    """Normalize raw µg/m³ values to [0, 1] using training-time MinMax ranges."""
    def _clip(val, lo, hi):
        return max(lo, min(val, hi))

    normed = [
        _clip(pm25, *MINMAX_RANGES["pm25"]) / MINMAX_RANGES["pm25"][1],
        _clip(pm10, *MINMAX_RANGES["pm10"]) / MINMAX_RANGES["pm10"][1],
        _clip(no2,  *MINMAX_RANGES["no2"])  / MINMAX_RANGES["no2"][1],
        _clip(o3,   *MINMAX_RANGES["o3"])   / MINMAX_RANGES["o3"][1],
        _clip(co,   *MINMAX_RANGES["co"])   / MINMAX_RANGES["co"][1],
    ]
    return np.array([normed])

MODEL_NAME = "air_quality_rf_corrected"
MODEL_PATH = os.path.join(os.path.dirname(__file__), f"{MODEL_NAME}.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "air_quality_scaler_corrected.pkl")

model = None
scaler = None

@app.on_event("startup")
def load_model():
    global model, scaler
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found at: {SCALER_PATH}")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and scaler loaded successfully.")

class AirQualityInput(BaseModel):
    pm25: float = Field(..., ge=0, description="PM2.5 concentration (µg/m³)")
    pm10: float = Field(..., ge=0, description="PM10 concentration (µg/m³)")
    no2: float = Field(..., ge=0, description="NO₂ concentration (µg/m³)")
    o3: float = Field(..., ge=0, description="O₃ concentration (µg/m³)")
    co: float = Field(..., ge=0, description="CO concentration (µg/m³)")

    class Config:
        json_schema_extra = {
            "example": {
                "pm25": 35.5,
                "pm10": 55.0,
                "no2": 40.0,
                "o3": 80.0,
                "co": 500.0,
            }
        }


class AirQualityBatchInput(BaseModel):
    readings: List[AirQualityInput] = Field(..., min_items=1, max_items=100)

    class Config:
        json_schema_extra = {
            "example": {
                "readings": [
                    {"pm25": 12.0, "pm10": 20.0, "no2": 15.0, "o3": 50.0, "co": 300.0},
                    {"pm25": 150.0, "pm10": 200.0, "no2": 100.0, "o3": 20.0, "co": 2000.0},
                ]
            }
        }


class PredictionResult(BaseModel):
    class_id: int
    class_label: str
    color: str
    health_advice: str
    probabilities: dict


class SinglePredictionResponse(BaseModel):
    input: AirQualityInput
    prediction: PredictionResult
    model: str = MODEL_NAME


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: List[SinglePredictionResponse]
    model: str = MODEL_NAME


#helper
def _predict_one(data: AirQualityInput) -> PredictionResult:
    #step 1 — MinMax normalize raw µg/m³ => [0, 1]  (same as training pipeline)
    normed = normalize_input(data.pm25, data.pm10, data.no2, data.o3, data.co)
    #step 2 — StandardScaler (fit on normalized training data)
    scaled = scaler.transform(normed)
    class_id = int(model.predict(scaled)[0])
    proba = model.predict_proba(scaled)[0]
    proba_dict = {
        CLASS_LABELS[i]: round(float(p), 4)
        for i, p in enumerate(proba)
        if i in CLASS_LABELS
    }
    return PredictionResult(
        class_id=class_id,
        class_label=CLASS_LABELS[class_id],
        color=CLASS_COLOR[class_id],
        health_advice=CLASS_ADVICE[class_id],
        probabilities=proba_dict,
    )


#endpoints
@app.get("/", tags=["Health"])
def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "api": "Air Quality Classification API",
        "version": "1.0.0",
        "model": "air_quality_rf_corrected",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    """Returns model loading status."""
    return {
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "feature_names": ["pm25", "pm10", "no2", "o3", "co"],
        "classes": CLASS_LABELS,
    }


@app.get("/classes", tags=["Info"])
def get_classes():
    """Returns all possible air quality classes with colors and advice."""
    return [
        {
            "class_id": k,
            "label": CLASS_LABELS[k],
            "color": CLASS_COLOR[k],
            "advice": CLASS_ADVICE[k],
        }
        for k in CLASS_LABELS
    ]


@app.post("/predict", response_model=SinglePredictionResponse, tags=["Prediction"])
def predict_single(data: AirQualityInput):
    """
    Classify air quality for a single set of pollutant readings.

    Returns the predicted AQI class, color code, health advice, and
    per-class probabilities.
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    result = _predict_one(data)
    return SinglePredictionResponse(input=data, prediction=result)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(payload: AirQualityBatchInput):
    """
    Classify air quality for a batch of pollutant readings (up to 100 rows).

    Returns results for each row in the same order as the input.
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    results = []
    for reading in payload.readings:
        pred = _predict_one(reading)
        results.append(SinglePredictionResponse(input=reading, prediction=pred))

    return BatchPredictionResponse(count=len(results), predictions=results)


@app.post("/predict/raw", tags=["Prediction"])
def predict_raw(data: AirQualityInput):
    """
    Returns only the class label (no probabilities) — lightweight endpoint.
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    normed = normalize_input(data.pm25, data.pm10, data.no2, data.o3, data.co)
    scaled = scaler.transform(normed)
    class_id = int(model.predict(scaled)[0])
    return {
        "class_id": class_id,
        "class_label": CLASS_LABELS[class_id],
        "color": CLASS_COLOR[class_id],
    }


#run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("air_quality_api:app", host="0.0.0.0", port=8000, reload=True)

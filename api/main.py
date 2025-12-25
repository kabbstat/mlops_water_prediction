from fastapi import FastAPI
from .schemas import WaterQuality, PredicitionResponse
import pandas as pd
from .model_loader import load_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title = "Water potability preidiction API",
              description="API to predict water potability using ML model",
              version="1.0.0")

model = None
@app.on_event("startup")
async def startup_event():
    """Load model when API starts"""
    global model, model_info
    try:
        model, model_info = load_model()
        logger.info(f"Model loaded successfully: {model_info}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.get("/")
def index():
    return {"message": "Welcome to the Water Potability Prediction API"}
@app.post("/predict", response_model= PredicitionResponse)
def predict_potability(data: WaterQuality):
    sample = pd.DataFrame({
        'ph': [data.ph],
        'Hardness': [data.Hardness],
        'Solids': [data.Solids],
        'Chloramines': [data.Chloramines],
        'Sulfate': [data.Sulfate],
        'Conductivity': [data.Conductivity],
        'Organic_carbon': [data.Organic_carbon],
        'Trihalomethanes': [data.Trihalomethanes],
        'Turbidity': [data.Turbidity]
    })
    prediction_value = model.predict(sample)[0]
    if prediction_value == 1:
        return {"prediction": "Potable"}
    else:
        return {"prediction": "Not Potable"}
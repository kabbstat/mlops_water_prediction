from pydantic import BaseModel
from typing import Optional
class WaterQuality(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float
class PredicitionResponse(BaseModel):
    prediction: str 
    probability: Optional[float] = None
    
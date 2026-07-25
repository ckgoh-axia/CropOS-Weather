from pydantic import BaseModel, Field
from typing import List
from datetime import datetime


class PredictRequest(BaseModel):
    farm_id: str
    lat: float = Field(..., ge=5.0, le=21.0, description="Farm latitude — Thailand bounds")
    lon: float = Field(..., ge=97.0, le=106.0, description="Farm longitude — Thailand bounds")
    timestamp: datetime


class HorizonForecast(BaseModel):
    horizon_hours: int
    rain_probability: float = Field(..., ge=0.0, le=1.0)
    alert: bool


class PredictResponse(BaseModel):
    farm_id: str
    forecast: List[HorizonForecast]
    model_version: str

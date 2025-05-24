from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from services.crypto_service import CryptoService
from typing import Dict
from enum import Enum

app = FastAPI(title="Data Science Term Project")

api_app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )

crypto_service = CryptoService()

class PredictionPeriod(str, Enum):
    ONE_HOUR = "1h"
    ONE_DAY = "1d"
    SEVEN_DAYS = "7d"

@api_app.get("/chart")
async def get_chart_data() -> Dict:
    try:
        return crypto_service.get_chart_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_app.get("/predict/{period}")
async def predict_price(period: PredictionPeriod) -> Dict:
    try:
        return crypto_service.predict_price(period.value)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_app.get("/fgindex")
async def get_fg_index() -> Dict:
    try:
        return crypto_service.get_fg_index()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.mount("/api", api_app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000
    ) 
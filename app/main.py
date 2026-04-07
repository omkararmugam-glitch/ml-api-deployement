from fastapi import FastAPI, HTTPException, status
from contextlib import asynccontextmanager

from app.schemas import (
    IrisFeatures, PredictionResponse,
    BatchRequest, BatchResponse,
    HealthResponse, ModelInfo
)
from app.model import ml_model, SPECIES, FEATURES, MODEL_VERSION


# Load model on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_model.load()
    yield


app = FastAPI(
    title="Iris Classification API",
    version=MODEL_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


# ROOT
@app.get("/")
def root():
    return {
        "message": "FINAL VERSION WORKING",
        "version": MODEL_VERSION
    }


# DEBUG (IMPORTANT)
@app.get("/check")
def check():
    return {"status": "deployment successful"}


# HEALTH
@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "healthy" if ml_model.is_loaded else "unhealthy",
        "model_loaded": ml_model.is_loaded,
        "version": MODEL_VERSION
    }


# MODEL INFO
@app.get("/model/info", response_model=ModelInfo)
def model_info():
    return {
        "name": "IrisClassifier",
        "version": MODEL_VERSION,
        "features": FEATURES,
        "classes": SPECIES
    }


# SINGLE PREDICTION
@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures):
    if not ml_model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    data = [
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]

    return ml_model.predict(data)


# BATCH PREDICTION
@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    if not ml_model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    batch = [
        [f.sepal_length, f.sepal_width, f.petal_length, f.petal_width]
        for f in request.instances
    ]

    results = ml_model.predict_batch(batch)

    return {
        "predictions": results,
        "count": len(results)
    }

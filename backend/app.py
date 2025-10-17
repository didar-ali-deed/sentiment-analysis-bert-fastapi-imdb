import os
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sentiment Analysis API", version="1.0.0")

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define paths
models_dir = "../models"
model_path = os.path.join(models_dir, "bert-imdb-final")

# Load model with device detection
device = 0 if torch.cuda.is_available() else -1
logger.info(f"Using device: {'GPU' if device >= 0 else 'CPU'}")

logger.info("Loading fine-tuned model and tokenizer...")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

logger.info("Creating prediction pipeline...")
sentiment_pipeline = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=device
)

class TextInput(BaseModel):
    text: str

label_map = {0: "Negative", 1: "Positive"}

@app.post("/predict/")
@limiter.limit("10/minute")  # Limit to 10 requests per minute
async def predict_sentiment(request: Request, input: TextInput):
    try:
        text = input.text.strip()
        if not text:
            logger.warning("Empty input received")
            raise HTTPException(status_code=400, detail="Text input cannot be empty")
        if len(text) > 512:
            logger.warning("Input too long")
            raise HTTPException(status_code=400, detail="Text input cannot exceed 512 characters")

        logger.info(f"Processing prediction for text: {text[:50]}...")
        prediction = sentiment_pipeline(text)[0]
        label = label_map[int(prediction["label"].split("_")[1])]
        score = prediction["score"]

        logger.info(f"Prediction: {label}, Score: {score}")
        return {
            "text": text,
            "predicted_label": label,
            "score": score
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Sentiment Analysis API is running!"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/model-info")
def model_info():
    return {
        "model": "BERT",
        "dataset": "IMDB",
        "version": "1.0.0",
        "device": "GPU" if device >= 0 else "CPU"
    }
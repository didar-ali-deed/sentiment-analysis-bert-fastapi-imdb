import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch

app = FastAPI()

# CORS for all origins in production (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific frontend URL after deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define paths
models_dir = "../models"  # Changed for deployment
model_path = os.path.join(models_dir, "bert-imdb-final")

# Load model with device detection
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device >= 0 else 'CPU'}")

print("Loading fine-tuned model and tokenizer...")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Creating prediction pipeline...")
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
async def predict_sentiment(input: TextInput):
    try:
        if not input.text.strip():
            raise HTTPException(status_code=400, detail="Text input cannot be empty")
        
        prediction = sentiment_pipeline(input.text)[0]
        label = label_map[int(prediction["label"].split("_")[1])]
        score = prediction["score"]
        
        return {
            "text": input.text,
            "predicted_label": label,
            "score": score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Sentiment Analysis API is running!"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}
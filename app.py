import threading
import subprocess
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer
import os
import schedule
import time
import logging
import requests

# Load environment variables from .env file
load_dotenv()

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

model_name = "models"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

class TranslationRequest(BaseModel):
    text: str

class TranslationsRequest(BaseModel):
    source_text: list[str]

class TrainingDataRequest(BaseModel):
    source_text: str
    target_text: str

class TrainingDatasRequest(BaseModel):
    data: list[TrainingDataRequest]
    

@app.post("/translate")
async def translate(request: TranslationRequest):
    inputs = tokenizer([">>vie<< " + request.text], return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    tgt_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return {"translation": tgt_text}

@app.post("/translates")
async def translate(request: TranslationsRequest):
    inputs = tokenizer([">>vie<< " + text for text in request.source_text], return_tensors="pt", padding=True)
    logger.info([">>vie<< " + text for text in request.source_text])
    translated = model.generate(**inputs)
    tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return {"translation": tgt_text}

@app.post("/add-training-data")
async def add_training_data(request: TrainingDataRequest):
    with open("data/train.source", "a", encoding="utf-8") as src_f, open("data/train.target", "a", encoding="utf-8") as tgt_f:
        src_f.write(request.source_text + "\n")
        tgt_f.write(request.target_text + "\n")
    return {"message": "Training data added successfully"}

@app.post("/add-training-datas")
async def add_training_data(request: TrainingDatasRequest):
    with open("data/train.source", "a", encoding="utf-8") as src_f, open("data/train.target", "a", encoding="utf-8") as tgt_f:
        src_f.write(request.source_text + "\n")
        tgt_f.write(request.target_text + "\n")
    return {"message": "Training data added successfully"}

@app.post("/reload-model")
async def reload_model():
    global tokenizer, model
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return {"message": "Model reloaded successfully"}

@app.get("/train-model")
async def train_model():
    train_model()
    return {"message": "Model training started"}

@app.get("/")
async def root():
    return {"message": "Translation API"}

def train_model():
    logger.info("Starting model training...")
    result = subprocess.run(["python", "scripts/training.py"], capture_output=True, text=True)
    logger.info(result.stdout)
    logger.info(result.stderr)
    logger.info("Model training completed.")
    # Tải lại mô hình sau khi huấn luyện
    response = requests.post("http://localhost:8000/reload-model")
    if response.status_code == 200:
        logger.info("Model reloaded successfully.")
    else:
        logger.error("Failed to reload model.")

def schedule_training():
    interval = int(os.getenv("TRAINING_INTERVAL", "60"))  # Interval in minutes
    logger.info(f"Scheduling model training every {interval} minutes.")
    schedule.every(interval).minutes.do(train_model)
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    training_thread = threading.Thread(target=schedule_training)
    training_thread.start()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
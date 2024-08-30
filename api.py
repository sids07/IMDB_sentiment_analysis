from pydantic import BaseModel
from fastapi import FastAPI
from inference import Inference
import argparse
import uvicorn

class RequestModel(BaseModel):
    text : str
    
app = FastAPI()

model_path = "model/model.pth"
tokenizer_path = "model/tokenizer.json"

inf = Inference(
    model_path = model_path,
    tokenizer_file_path = tokenizer_path
)

@app.post("/get_sentiment")
def get_response(
    data: RequestModel,
):  
    text = data.text
    
    return {
        "sentiment": inf.get_sentiment(text)
    }
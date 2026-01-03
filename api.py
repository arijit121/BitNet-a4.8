import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from loguru import logger
import os

from main import BitNetModel, BitNetConfig
from model_wrapper import BitNetForCausalLM, SimpleTokenizer

# --- Data Models ---
class ChatRequest(BaseModel):
    message: str
    max_length: int = 100
    temperature: float = 1.0

class ChatResponse(BaseModel):
    response: str

# --- Application Setup ---
app = FastAPI(title="BitNet Chatbot API")

logger.info("Initializing model...")
tokenizer = SimpleTokenizer()

# Use the centralized Medium config
from model_wrapper import get_training_config
config = get_training_config(tokenizer.vocab_size)

model = BitNetForCausalLM(config)

# Load trained weights if they exist
model_path = "bitnet_model.pth"
if os.path.exists(model_path):
    logger.info(f"Loading trained model from {model_path}")
    try:
        model.load_state_dict(torch.load(model_path))
    except Exception as e:
         logger.warning(f"Could not load weights: {e}. Using random weights.")
else:
    logger.warning("No trained model found. Using random weights.")

logger.info("Model initialized.")

@app.get("/")
async def health_check():
    model_status = "training_params_loaded" if os.path.exists(model_path) else "random_weights"
    return {
        "status": "ok",
        "model": "BitNet-a4.8",
        "model_status": model_status,
        "config": {
            "hidden_size": config.hidden_size,
            "layers": config.num_hidden_layers,
            "vocab_size": config.vocab_size
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    logger.info(f"Received message: {request.message}")
    
    try:
        # Tokenize
        input_ids = torch.tensor([tokenizer.encode(request.message)], dtype=torch.long)
        
        # Generate
        output_ids = model.generate(
            input_ids, 
            max_new_tokens=request.max_length, 
            temperature=request.temperature
        )
        
        # Decode - skip prompt
        generated_ids = output_ids[0].tolist()[input_ids.shape[1]:]
        response_text = tokenizer.decode(generated_ids)
        
        logger.info(f"Generated response: {response_text}")
        return ChatResponse(response=response_text)
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

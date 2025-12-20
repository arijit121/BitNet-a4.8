import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import time
from loguru import logger
import csv

from model_wrapper import BitNetForCausalLM, SimpleTokenizer, get_training_config

# --- Config ---
BATCH_SIZE = 16
BLOCK_SIZE = 128   # Sequence length for training
EPOCHS = 3
LEARNING_RATE = 1e-3
DEVICE = "cpu"      # Force CPU for now as requested (or auto-detect if CUDA available, but keep it simple)

# --- Dataset ---
class TextDataset(Dataset):
    def __init__(self, text, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.data = tokenizer.encode(text)
        self.block_size = block_size
        
    def __len__(self):
        # Use stride = block_size to avoid huge overlap and massive epoch size
        return (len(self.data) - 1) // self.block_size

    def __getitem__(self, idx):
        # Start index for this chunk
        start_idx = idx * self.block_size
        chunk = self.data[start_idx : start_idx + self.block_size + 1]
        
        # Pad if short (shouldn't happen often with drop_last, but good to be safe)
        if len(chunk) < self.block_size + 1:
            chunk = chunk + [self.tokenizer.pad_token_id] * (self.block_size + 1 - len(chunk))
            
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def load_csv_data(filepath):
    logger.info(f"Loading data from {filepath}...")
    text_data = ""
    count = 0
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Format: Question\nOptions\nAnswer
                # Ensure all fields exist
                if not row.get('prompt') or not row.get('answer'):
                    continue
                    
                entry = f"Question: {row['prompt']}\n"
                entry += "Options:\n"
                entry += f"(A) {row.get('A', '')}\n"
                entry += f"(B) {row.get('B', '')}\n"
                entry += f"(C) {row.get('C', '')}\n"
                entry += f"(D) {row.get('D', '')}\n"
                entry += f"(E) {row.get('E', '')}\n"
                entry += f"Answer: {row['answer']}\n\n"
                
                text_data += entry
                count += 1
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        return ""
        
    logger.info(f"Loaded {count} examples from CSV.")
    return text_data

def train():
    logger.info(f"Using device: {DEVICE}")
    
    # 1. Load Data
    input_file = "TrainDataset.csv"
    if not os.path.exists(input_file):
        logger.error(f"File {input_file} not found.")
        return

    text = load_csv_data(input_file)
        
    if len(text) < BLOCK_SIZE + 1:
        logger.error(f"Text data is too short. Loaded {len(text)} chars.")
        return

    # 2. Setup Model & Tokenizer
    tokenizer = SimpleTokenizer()
    dataset = TextDataset(text, tokenizer, BLOCK_SIZE)
    # Use drop_last=True to avoid partial batches that might cause issues if very small
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Use the centralized Medium config
    config = get_training_config(tokenizer.vocab_size)
    
    model = BitNetForCausalLM(config)
    model.to(DEVICE)
    
    # Check for existing checkpoint
    checkpoint_path = "bitnet_model.pth"
    if os.path.exists(checkpoint_path):
        try:
            state_dict = torch.load(checkpoint_path)
            # Check if sizes match roughly (simple heuristic by checking a known key size)
            if state_dict['model.embed_tokens.weight'].shape[1] != config.hidden_size:
                 logger.warning("Checkpoint architecture mismatch (hidden size differs). Starting fresh.")
                 # Backup old model
                 os.rename(checkpoint_path, "bitnet_model_old.pth")
            else:
                model.load_state_dict(state_dict)
                logger.info("Resuming from existing checkpoint.")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}. Starting fresh.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 3. Training Loop
    model.train()
    total_tokens = len(text)
    logger.info(f"Starting training on {total_tokens} characters ({len(dataset)} sequences)...")
    logger.info(f"Config: Hidden={config.hidden_size}, Layers={config.num_hidden_layers}")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        start_time = time.time()
        
        # Limit batches for demo speed if dataset is huge, but let's try full
        # If it's too slow, the user can stop it.
        
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(x)
            
            # Reshape for loss: (batch * seq, vocab) vs (batch * seq)
            loss = criterion(logits.view(-1, config.vocab_size), y.view(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.4f} | Time: {time.time()-start_time:.2f}s")
        
        # Save checkpoint every epoch
        torch.save(model.state_dict(), "bitnet_model.pth")

    logger.info("Training complete. Model saved to bitnet_model.pth")

if __name__ == "__main__":
    train()

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from main import BitNetModel, BitNetConfig

# --- Simple Tokenizer ---
class SimpleTokenizer:
    """A basic character-level tokenizer for demonstration."""
    def __init__(self):
        # Basic ASCII chars + some common special chars
        chars = sorted(list(set(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-_\n\r'\"()[]:;")))
        self.vocab_size = len(chars) + 3 # +3 for pad, bos, eos
        self.char_to_id = {c: i + 3 for i, c in enumerate(chars)}
        self.id_to_char = {i + 3: c for i, c in enumerate(chars)}
        
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        
    def encode(self, text: str) -> List[int]:
        ids = [self.char_to_id.get(c, self.pad_token_id) for c in text]
        return [self.bos_token_id] + ids
        
    def decode(self, ids: List[int]) -> str:
        text = ""
        for idx in ids:
            if idx == self.eos_token_id:
                break
            if idx in [self.pad_token_id, self.bos_token_id]:
                continue
            text += self.id_to_char.get(idx, "")
        return text

# --- Model Wrapper ---
class BitNetForCausalLM(nn.Module):
    """Wrapper around BitNetModel to add a language model head."""
    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.model = BitNetModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.model(input_ids, attention_mask)
        logits = self.lm_head(hidden_states)
        return logits
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50, temperature: float = 1.0) -> torch.Tensor:
        """Simple greedy/temperature sampling generation loop."""
        self.eval()
        curr_ids = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self(curr_ids)
            next_token_logits = outputs[:, -1, :] / temperature
            
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            
            # Stop if EOS (optional, not strictly enforced here for simplicity)
            if next_token.item() == 2: # EOS
                break
                
        return curr_ids

def get_training_config(vocab_size: int):
    """
    Returns the 'Medium' configuration for BitNet, suitable for 12GB RAM.
    """
    return BitNetConfig(
        vocab_size=vocab_size,
        hidden_size=1024,
        intermediate_size=2048,
        num_hidden_layers=8,
        num_attention_heads=8,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        max_position_embeddings=512
    )

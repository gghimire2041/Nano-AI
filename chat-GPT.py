"""
Mini ChatGPT: A Miniature Language Model Implementation
=====================================================

This is a simplified but functional implementation of a ChatGPT-like model
that demonstrates the core concepts of transformer architecture and language modeling.

Features:
- Simple transformer architecture with attention mechanism
- Character-level tokenization for simplicity
- Training loop with loss monitoring
- Text generation with temperature control
- Interactive chat interface

Usage:
1. Run this script
2. The model will train on sample data
3. You can then chat with your mini model!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
from typing import List, Tuple
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class MultiHeadAttention(nn.Module):
    """Simplified multi-head attention mechanism"""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # Linear projections
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply causal mask (for autoregressive generation)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads and put through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        return self.w_o(attn_output)

class FeedForward(nn.Module):
    """Simple feed-forward network"""
    
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Single transformer block with attention and feed-forward"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x

class MiniChatGPT(nn.Module):
    """Miniature ChatGPT model"""
    
    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4, 
                 n_layers: int = 4, max_seq_len: int = 128):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 4)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, targets=None):
        batch_size, seq_len = x.size()
        
        # Create position indices
        pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(batch_size, seq_len)
        
        # Embeddings
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(pos)
        x = self.dropout(tok_emb + pos_emb)
        
        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # Calculate cross-entropy loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss

class SimpleTokenizer:
    """Character-level tokenizer for simplicity"""
    
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        
    def fit(self, text: str):
        """Build vocabulary from text"""
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
    def encode(self, text: str) -> List[int]:
        """Convert text to token indices"""
        return [self.char_to_idx.get(ch, 0) for ch in text]
    
    def decode(self, tokens: List[int]) -> str:
        """Convert token indices back to text"""
        return ''.join([self.idx_to_char.get(tok, '<UNK>') for tok in tokens])

class MiniChatGPTTrainer:
    """Training utilities for the mini model"""
    
    def __init__(self, model: MiniChatGPT, tokenizer: SimpleTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def prepare_data(self, text: str, seq_len: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare training data from text"""
        tokens = self.tokenizer.encode(text)
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(0, len(tokens) - seq_len, seq_len // 2):  # Overlapping sequences
            seq = tokens[i:i + seq_len]
            target = tokens[i + 1:i + seq_len + 1]
            
            if len(seq) == seq_len and len(target) == seq_len:
                sequences.append(seq)
                targets.append(target)
        
        return (torch.tensor(sequences, device=self.device), 
                torch.tensor(targets, device=self.device))
    
    def train(self, training_text: str, epochs: int = 100, lr: float = 0.001):
        """Train the model"""
        print(f"Training on {self.device}")
        print(f"Training text length: {len(training_text)} characters")
        
        # Prepare data
        X, y = self.prepare_data(training_text)
        print(f"Created {len(X)} training sequences")
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            
            # Shuffle data
            indices = torch.randperm(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(0, len(X_shuffled), 8):  # Batch size of 8
                batch_X = X_shuffled[i:i+8]
                batch_y = y_shuffled[i:i+8]
                
                optimizer.zero_grad()
                logits, loss = self.model(batch_X, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / (len(X_shuffled) // 8)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}, Loss: {avg_loss:.4f}")
    
    def generate(self, prompt: str, max_length: int = 100, temperature: float = 0.8) -> str:
        """Generate text from a prompt"""
        self.model.eval()
        
        # Encode prompt
        tokens = self.tokenizer.encode(prompt)
        tokens = torch.tensor(tokens, device=self.device).unsqueeze(0)
        
        generated = tokens.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                logits, _ = self.model(generated)
                
                # Get logits for the last token
                logits = logits[0, -1, :] / temperature
                
                # Apply softmax and sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                
                # Stop if we've generated enough or hit max context
                if generated.size(1) >= self.model.max_seq_len:
                    break
        
        # Decode generated tokens
        generated_tokens = generated[0].tolist()
        return self.tokenizer.decode(generated_tokens)

def create_sample_data():
    """Create sample training data"""
    return """
Hello! How are you doing today?
I'm doing great, thank you for asking!
What can I help you with?
I'd like to learn about artificial intelligence.
AI is fascinating! It involves creating systems that can think and learn.
That sounds interesting! Can you tell me more?
Sure! AI includes machine learning, neural networks, and natural language processing.
What is machine learning?
Machine learning is a subset of AI where computers learn from data.
How does that work?
Computers find patterns in data and use them to make predictions.
That's amazing! What about neural networks?
Neural networks are inspired by how the brain works.
They consist of connected nodes that process information.
This is really cool! I want to learn more about this.
Great! There's so much to explore in AI and machine learning.
What other topics interest you?
I'm curious about programming and computer science.
Programming is the art of giving instructions to computers.
You can create amazing things with code!
What programming language should I start with?
Python is great for beginners! It's easy to read and very powerful.
Thanks for the advice! This has been very helpful.
You're welcome! Feel free to ask if you have more questions.
Have a great day!
You too! Keep learning and exploring!
"""

def main():
    """Main function to run the mini ChatGPT demo"""
    print("🤖 Mini ChatGPT - Building Your Own Language Model!")
    print("=" * 55)
    
    # Create sample data
    training_data = create_sample_data()
    
    # Initialize tokenizer
    print("📝 Setting up tokenizer...")
    tokenizer = SimpleTokenizer()
    tokenizer.fit(training_data)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create model
    print("🧠 Creating mini model...")
    model = MiniChatGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=64,      # Smaller for demo
        n_heads=4,
        n_layers=3,
        max_seq_len=64
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = MiniChatGPTTrainer(model, tokenizer)
    
    # Train the model
    print("\n🚀 Training the model...")
    trainer.train(training_data, epochs=50, lr=0.001)
    
    print("\n✅ Training completed!")
    
    # Interactive chat
    print("\n💬 Chat with your mini model! (type 'quit' to exit)")
    print("-" * 40)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Generate response
        print("Mini GPT: ", end="", flush=True)
        response = trainer.generate(user_input, max_length=50, temperature=0.7)
        
        # Extract just the generated part (after the prompt)
        generated_part = response[len(user_input):]
        
        # Clean up the response (stop at natural break points)
        for stop_char in ['\n', '!', '?', '.']:
            if stop_char in generated_part:
                generated_part = generated_part.split(stop_char)[0] + stop_char
                break
        
        print(generated_part)

if __name__ == "__main__":
    main()

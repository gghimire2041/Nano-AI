"""
Flask API Server to serve the actual Python AI models
====================================================

This creates a web API that your HTML can call to get real responses
from the actual neural networks (including the gibberish outputs).

Install requirements:
pip install flask flask-cors torch

Run this file:
python app.py

Then your HTML will get real AI model responses!
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
from typing import List, Tuple, Optional

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from your HTML

class SimpleTokenizer:
    """Character-level tokenizer for all models"""
    
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        
    def fit(self, text: str):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
    def encode(self, text: str) -> List[int]:
        return [self.char_to_idx.get(ch, 0) for ch in text]
    
    def decode(self, tokens: List[int]) -> str:
        return ''.join([self.idx_to_char.get(tok, '<UNK>') for tok in tokens])

# Copy your AI model classes here (ChatGPTAttention, MiniChatGPT, etc.)
class ChatGPTAttention(nn.Module):
    """ChatGPT-style attention with pre-layer norm"""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_qkv = nn.Linear(d_model, 3 * d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        qkv = self.w_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2) for t in qkv]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.w_o(attn_output)

class ChatGPTBlock(nn.Module):
    """ChatGPT transformer block with pre-layer normalization"""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = ChatGPTAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        
    def forward(self, x, mask=None):
        x = x + self.attention(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x

class MiniChatGPT(nn.Module):
    """Mini ChatGPT with RLHF simulation"""
    
    def __init__(self, vocab_size: int, d_model: int = 32, n_heads: int = 2, n_layers: int = 2, max_seq_len: int = 32):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.blocks = nn.ModuleList([ChatGPTBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, targets=None):
        batch_size, seq_len = x.size()
        
        pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.position_embedding(pos)
        
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)
        
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss

# Add other model classes (MiniGrok, MiniGemini, MiniDeepSeek, MiniClaude) here...
# For brevity, I'll just show ChatGPT, but you'd include all 5 models

class UniversalTrainer:
    """Universal trainer for all mini models"""
    
    def __init__(self, model: nn.Module, tokenizer: SimpleTokenizer, model_name: str):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def prepare_data(self, text: str, seq_len: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer.encode(text)
        sequences, targets = [], []
        
        for i in range(0, len(tokens) - seq_len, seq_len // 2):
            seq = tokens[i:i + seq_len]
            target = tokens[i + 1:i + seq_len + 1]
            if len(seq) == seq_len and len(target) == seq_len:
                sequences.append(seq)
                targets.append(target)
        
        return (torch.tensor(sequences, device=self.device), 
                torch.tensor(targets, device=self.device))
    
    def train(self, training_text: str, epochs: int = 20, lr: float = 0.002):
        """Quick training for demo purposes"""
        print(f"🚀 Training {self.model_name}...")
        
        X, y = self.prepare_data(training_text)
        if len(X) == 0:
            print(f"Not enough data for {self.model_name}")
            return
            
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            
            indices = torch.randperm(len(X))
            X_shuffled, y_shuffled = X[indices], y[indices]
            
            for i in range(0, len(X_shuffled), 2):
                batch_X = X_shuffled[i:i+2]
                batch_y = y_shuffled[i:i+2]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X, batch_y)
                loss = outputs[1] if isinstance(outputs, tuple) else outputs
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 5 == 0:
                avg_loss = total_loss / (len(X_shuffled) // 2)
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
    
    def generate(self, prompt: str, max_length: int = 30, temperature: float = 0.8) -> str:
        """Generate text from a prompt - THIS IS THE REAL AI OUTPUT"""
        self.model.eval()
        
        tokens = self.tokenizer.encode(prompt)
        tokens = torch.tensor(tokens, device=self.device).unsqueeze(0)
        generated = tokens.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.model(generated)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                
                logits = logits[0, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                
                if generated.size(1) >= self.model.max_seq_len:
                    break
        
        # THIS IS THE REAL NEURAL NETWORK OUTPUT (might be gibberish!)
        return self.tokenizer.decode(generated[0].tolist())

def create_training_data():
    """Create training data for each model"""
    return {
        'chatgpt': "Hello! How can I help you today? I'm here to assist with any questions you have. What would you like to know about? I can help with explanations, writing, and problem-solving. Feel free to ask me anything!",
        'grok': "Let me think about this differently. Here's an unconventional perspective on that question. That's an interesting way to look at it! I love exploring creative solutions to problems. What if we approached this from a totally different angle?",
        'gemini': "I can help you with text, images, and multimodal tasks. Let me analyze this comprehensively for you. Here's what I understand from your input. I can process different types of information together. Would you like me to explain this step by step?",
        'deepseek': "Let me process this through my specialized reasoning modules. I'll apply focused expertise to solve this problem. Using advanced reasoning capabilities for this task. Let me break this down systematically. Here's my detailed analysis of the situation.",
        'claude': "I want to be helpful, harmless, and honest in my response. Let me make sure my answer is both accurate and safe. I'll be thoughtful about the implications of this topic. Is there anything concerning about this request I should address? I aim to provide balanced and ethical guidance."
    }

# Initialize models and tokenizer
print("🤖 Initializing AI Models...")
tokenizer = SimpleTokenizer()
training_data = create_training_data()

# Fit tokenizer on all data
all_text = ' '.join(training_data.values())
tokenizer.fit(all_text)

# Initialize models
models = {
    'chatgpt': MiniChatGPT(tokenizer.vocab_size, d_model=32, n_heads=2, n_layers=2, max_seq_len=32)
    # Add other models here: MiniGrok, MiniGemini, MiniDeepSeek, MiniClaude
}

trainers = {}

# Train each model
for name, model in models.items():
    trainer = UniversalTrainer(model, tokenizer, name)
    trainer.train(training_data[name], epochs=20, lr=0.002)
    trainers[name] = trainer

print("✅ All models trained and ready!")

@app.route('/chat', methods=['POST'])
def chat():
    """API endpoint that returns REAL AI model responses"""
    try:
        data = request.json
        model_type = data.get('model', 'chatgpt')
        message = data.get('message', 'hello')
        
        if model_type not in trainers:
            return jsonify({'error': f'Model {model_type} not available'}), 400
        
        # Get REAL neural network response (might be gibberish!)
        trainer = trainers[model_type]
        real_response = trainer.generate(message, max_length=25, temperature=0.7)
        
        # Extract just the generated part (after the prompt)
        if message.lower() in real_response.lower():
            # Find where the original prompt ends
            prompt_end = real_response.lower().find(message.lower()) + len(message)
            generated_part = real_response[prompt_end:].strip()
        else:
            generated_part = real_response
        
        return jsonify({
            'response': generated_part,
            'model': model_type,
            'full_output': real_response
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'OK', 'models': list(trainers.keys())})

if __name__ == '__main__':
    print("🚀 Starting Flask API server...")
    print("💡 Your HTML can now get real AI responses at http://localhost:5000/chat")
    app.run(debug=True, host='0.0.0.0', port=5000)

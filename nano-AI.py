"""
Mini AI Models Collection
========================

Miniature implementations of major AI models demonstrating their unique architectures:
- ChatGPT (OpenAI): Decoder-only transformer with RLHF
- Grok (xAI): Mixture of Experts (MoE) architecture
- Gemini (Google): Multimodal transformer with advanced attention
- DeepSeek (DeepSeek): MoE with expert routing optimizations
- Claude (Anthropic): Constitutional AI with harmlessness training

Each model demonstrates the core architectural principles of the actual systems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
from typing import List, Tuple, Optional, Dict, Any
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

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

# ============================================================================
# 1. MINI CHATGPT (OpenAI) - Decoder-only Transformer with RLHF simulation
# ============================================================================

class ChatGPTAttention(nn.Module):
    """ChatGPT-style attention with pre-layer norm"""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_qkv = nn.Linear(d_model, 3 * d_model)  # Combined QKV projection
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # Combined QKV projection (ChatGPT optimization)
        qkv = self.w_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2) for t in qkv]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Concatenate heads
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
            nn.GELU(),  # ChatGPT uses GELU activation
            nn.Linear(4 * d_model, d_model)
        )
        
    def forward(self, x, mask=None):
        # Pre-layer norm (GPT-2 style, used in ChatGPT)
        x = x + self.attention(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x

class MiniChatGPT(nn.Module):
    """Mini ChatGPT with RLHF simulation"""
    
    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 3, max_seq_len: int = 64):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.blocks = nn.ModuleList([ChatGPTBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # RLHF components (simplified)
        self.reward_head = nn.Linear(d_model, 1)  # For reward modeling
        
    def forward(self, x, targets=None, return_rewards=False):
        batch_size, seq_len = x.size()
        
        pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.position_embedding(pos)
        
        # Causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)
        
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        if return_rewards:
            rewards = self.reward_head(x).squeeze(-1)
            return logits, loss, rewards
        
        return logits, loss

# ============================================================================
# 2. MINI GROK (xAI) - Mixture of Experts Architecture
# ============================================================================

class MoEGate(nn.Module):
    """Mixture of Experts gating mechanism"""
    
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts)
        
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.size()
        x_flat = x.view(-1, d_model)
        
        gate_logits = self.gate(x_flat)  # (batch_size * seq_len, num_experts)
        
        # Top-k gating
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        
        return top_k_weights, top_k_indices

class MoELayer(nn.Module):
    """Mixture of Experts layer (Grok's key innovation)"""
    
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.gate = MoEGate(d_model, num_experts, top_k)
        
        # Multiple expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        x_flat = x.view(-1, d_model)
        
        # Get gating decisions
        gate_weights, expert_indices = self.gate(x)
        
        # Process through selected experts
        output = torch.zeros_like(x_flat)
        
        for i, expert in enumerate(self.experts):
            # Find tokens that should use this expert
            expert_mask = (expert_indices == i).any(dim=-1)
            if expert_mask.any():
                expert_input = x_flat[expert_mask]
                expert_output = expert(expert_input)
                
                # Weight the expert output
                for j, token_idx in enumerate(torch.where(expert_mask)[0]):
                    if i in expert_indices[token_idx]:
                        weight_idx = (expert_indices[token_idx] == i).nonzero()[0, 0]
                        weight = gate_weights[token_idx, weight_idx]
                        output[token_idx] += weight * expert_output[j]
        
        return output.view(batch_size, seq_len, d_model)

class GrokBlock(nn.Module):
    """Grok transformer block with MoE"""
    
    def __init__(self, d_model: int, n_heads: int, num_experts: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = ChatGPTAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.moe = MoELayer(d_model, d_model * 4, num_experts)
        
    def forward(self, x, mask=None):
        x = x + self.attention(self.ln1(x), mask)
        x = x + self.moe(self.ln2(x))  # MoE instead of regular FFN
        return x

class MiniGrok(nn.Module):
    """Mini Grok with Mixture of Experts"""
    
    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 3, 
                 max_seq_len: int = 64, num_experts: int = 4):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.blocks = nn.ModuleList([
            GrokBlock(d_model, n_heads, num_experts) for _ in range(n_layers)
        ])
        
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

# ============================================================================
# 3. MINI GEMINI (Google) - Multimodal Transformer with Advanced Attention
# ============================================================================

class GeminiAttention(nn.Module):
    """Gemini-style multi-query attention"""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Multi-query attention: shared K,V across heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, self.d_k)  # Shared across heads
        self.w_v = nn.Linear(d_model, self.d_k)  # Shared across heads
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # Multi-query: Q has multiple heads, K,V are shared
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).unsqueeze(1).expand(-1, self.n_heads, -1, -1)  # Broadcast to all heads
        V = self.w_v(x).unsqueeze(1).expand(-1, self.n_heads, -1, -1)  # Broadcast to all heads
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.w_o(attn_output)

class GeminiBlock(nn.Module):
    """Gemini transformer block with advanced features"""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = GeminiAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        
        # SwiGLU activation (used in Gemini)
        self.w_gate = nn.Linear(d_model, d_model * 4)
        self.w_up = nn.Linear(d_model, d_model * 4)
        self.w_down = nn.Linear(d_model * 4, d_model)
        
    def forward(self, x, mask=None):
        x = x + self.attention(self.ln1(x), mask)
        
        # SwiGLU activation
        residual = x
        x = self.ln2(x)
        gate = F.silu(self.w_gate(x))  # SiLU activation
        up = self.w_up(x)
        x = self.w_down(gate * up)
        
        return residual + x

class MiniGemini(nn.Module):
    """Mini Gemini with multimodal capabilities (text-only demo)"""
    
    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 3, max_seq_len: int = 64):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Gemini uses RoPE (Rotary Position Encoding) - simplified version
        self.rotary_emb = nn.Parameter(torch.randn(max_seq_len, d_model))
        
        self.blocks = nn.ModuleList([GeminiBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, targets=None):
        batch_size, seq_len = x.size()
        
        pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.position_embedding(pos)
        
        # Add rotary positional encoding
        x = x + self.rotary_emb[:seq_len].unsqueeze(0)
        
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)
        
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss

# ============================================================================
# 4. MINI DEEPSEEK (DeepSeek) - Optimized MoE with Expert Routing
# ============================================================================

class DeepSeekMoE(nn.Module):
    """DeepSeek's optimized MoE with better expert routing"""
    
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Improved gating with load balancing
        self.gate = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.SiLU(),  # DeepSeek uses SiLU
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])
        
        # Load balancing auxiliary loss
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        x_flat = x.view(-1, d_model)
        
        gate_logits = self.gate(x_flat)
        
        # Add noise for better expert utilization (DeepSeek innovation)
        if self.training:
            noise = torch.randn_like(gate_logits) * 0.1
            gate_logits = gate_logits + noise
        
        # Top-k selection with load balancing
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        
        # Update expert usage counts for load balancing
        if self.training:
            for i in range(self.num_experts):
                count = (top_k_indices == i).sum().float()
                self.expert_counts[i] = 0.99 * self.expert_counts[i] + 0.01 * count
        
        # Process through experts
        output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            expert_mask = (top_k_indices == i).any(dim=-1)
            if expert_mask.any():
                expert_input = x_flat[expert_mask]
                expert_output = expert(expert_input)
                
                for j, token_idx in enumerate(torch.where(expert_mask)[0]):
                    if i in top_k_indices[token_idx]:
                        weight_idx = (top_k_indices[token_idx] == i).nonzero()[0, 0]
                        weight = top_k_weights[token_idx, weight_idx]
                        output[token_idx] += weight * expert_output[j]
        
        return output.view(batch_size, seq_len, d_model)

class DeepSeekBlock(nn.Module):
    """DeepSeek transformer block"""
    
    def __init__(self, d_model: int, n_heads: int, num_experts: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = GeminiAttention(d_model, n_heads)  # Uses multi-query attention
        self.ln2 = nn.LayerNorm(d_model)
        self.moe = DeepSeekMoE(d_model, d_model * 4, num_experts)
        
    def forward(self, x, mask=None):
        x = x + self.attention(self.ln1(x), mask)
        x = x + self.moe(self.ln2(x))
        return x

class MiniDeepSeek(nn.Module):
    """Mini DeepSeek with optimized MoE"""
    
    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 3, 
                 max_seq_len: int = 64, num_experts: int = 4):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.blocks = nn.ModuleList([
            DeepSeekBlock(d_model, n_heads, num_experts) for _ in range(n_layers)
        ])
        
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

# ============================================================================
# 5. MINI CLAUDE (Anthropic) - Constitutional AI with Harmlessness Training
# ============================================================================

class ClaudeBlock(nn.Module):
    """Claude transformer block with Constitutional AI features"""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = ChatGPTAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Claude uses GLU variants
        self.w1 = nn.Linear(d_model, d_model * 4)
        self.w2 = nn.Linear(d_model * 4, d_model)
        self.w3 = nn.Linear(d_model, d_model * 4)  # For gating
        
    def forward(self, x, mask=None):
        x = x + self.attention(self.ln1(x), mask)
        
        # GLU activation
        residual = x
        x = self.ln2(x)
        gate = F.sigmoid(self.w3(x))  # Gating mechanism
        x = self.w2(F.silu(self.w1(x)) * gate)
        
        return residual + x

class MiniClaude(nn.Module):
    """Mini Claude with Constitutional AI simulation"""
    
    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 3, max_seq_len: int = 64):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.blocks = nn.ModuleList([ClaudeBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # Constitutional AI components
        self.harmfulness_head = nn.Linear(d_model, 2)  # Harmful vs Harmless
        self.helpfulness_head = nn.Linear(d_model, 2)  # Helpful vs Not helpful
        
    def forward(self, x, targets=None, return_constitutional_scores=False):
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
        
        if return_constitutional_scores:
            # Constitutional AI scores
            harmfulness_scores = F.softmax(self.harmfulness_head(x), dim=-1)
            helpfulness_scores = F.softmax(self.helpfulness_head(x), dim=-1)
            return logits, loss, harmfulness_scores, helpfulness_scores
        
        return logits, loss

# ============================================================================
# UNIVERSAL TRAINER FOR ALL MODELS
# ============================================================================

class UniversalTrainer:
    """Universal trainer for all mini models"""
    
    def __init__(self, model: nn.Module, tokenizer: SimpleTokenizer, model_name: str):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def prepare_data(self, text: str, seq_len: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
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
    
    def train(self, training_text: str, epochs: int = 30, lr: float = 0.001):
        print(f"\n🚀 Training {self.model_name} on {self.device}")
        
        X, y = self.prepare_data(training_text)
        print(f"Training sequences: {len(X)}")
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            
            indices = torch.randperm(len(X))
            X_shuffled, y_shuffled = X[indices], y[indices]
            
            for i in range(0, len(X_shuffled), 4):  # Smaller batch size
                batch_X = X_shuffled[i:i+4]
                batch_y = y_shuffled[i:i+4]
                
                optimizer.zero_grad()
                
                # Handle different model outputs
                outputs = self.model(batch_X, batch_y)
                if isinstance(outputs, tuple):
                    loss = outputs[1]  # All models return loss as second element
                else:
                    loss = outputs
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                avg_loss = total_loss / (len(X_shuffled) // 4)
                print(f"Epoch {epoch:2d}, Loss: {avg_loss:.4f}")
    
    def generate(self, prompt: str, max_length: int = 50, temperature: float = 0.8) -> str:
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
        
        return self.tokenizer.decode(generated[0].tolist())

def create_training_data():
    """Create specialized training data for each model type"""
    return {
        'chatgpt': """
Hello! How can I help you today?
I'm here to assist with any questions you have.
What would you like to know about?
I can help with explanations, writing, and problem-solving.
Feel free to ask me anything!
""",
        'grok': """
Let me think about this differently.
Here's an unconventional perspective on that question.
That's an interesting way to look at it!
I love exploring creative solutions to problems.
What if we approached this from a totally different angle?
""",
        'gemini': """
I can help you with text, images, and multimodal tasks.
Let me analyze this comprehensively for you.
Here's what I understand from your input.
I can process different types of information together.
Would you like me to explain this step by step?
""",
        'deepseek': """
Let me process this through my specialized reasoning modules.
I'll apply focused expertise to solve this problem.
Using advanced reasoning capabilities for this task.
Let me break this down systematically.
Here's my detailed analysis of the situation.
""",
        'claude': """
I want to be helpful, harmless, and honest in my response.
Let me make sure my answer is both accurate and safe.
I'll be thoughtful about the implications of this topic.
Is there anything concerning about this request I should address?
I aim to provide balanced and ethical guidance.
"""
    }

def main():
    """Demonstrate all mini AI models"""
    
    print("🤖 Mini AI Models Collection")
    print("=" * 50)
    print("Demonstrating miniature versions of:")
    print("• ChatGPT (OpenAI) - Decoder transformer + RLHF")
    print("• Grok (xAI) - Mixture of Experts")
    print("• Gemini (Google) - Multimodal transformer")
    print("• DeepSeek - Optimized MoE")
    print("• Claude (Anthropic) - Constitutional AI")
    print()
    
    # Create training data
    training_data = create_training_data()
    
    # Initialize tokenizer with all data
    all_text = ' '.join(training_data.values())
    tokenizer = SimpleTokenizer()
    tokenizer.fit(all_text)
    print(f"📝 Vocabulary size: {tokenizer.vocab_size}")
    
    # Model configurations
    models = {
        'ChatGPT': MiniChatGPT(tokenizer.vocab_size, d_model=32, n_heads=2, n_layers=2),
        'Grok': MiniGrok(tokenizer.vocab_size, d_model=32, n_heads=2, n_layers=2, num_experts=2),
        'Gemini': MiniGemini(tokenizer.vocab_size, d_model=32, n_heads=2, n_layers=2),
        'DeepSeek': MiniDeepSeek(tokenizer.vocab_size, d_model=32, n_heads=2, n_layers=2, num_experts=2),
        'Claude': MiniClaude(tokenizer.vocab_size, d_model=32, n_heads=2, n_layers=2)
    }
    
    trainers = {}
    
    # Train each model
    for name, model in models.items():
        print(f"\n🔥 Training Mini {name}")
        trainer = UniversalTrainer(model, tokenizer, name)
        
        # Use specific training data for each model
        data_key = name.lower()
        trainer.train(training_data[data_key], epochs=20, lr=0.002)
        trainers[name] = trainer
        
        print(f"✅ {name} training completed!")
    
    # Interactive comparison
    print("\n" + "="*50)
    print("🎭 Compare AI Models - Same Prompt, Different Responses!")
    print("="*50)
    
    while True:
        prompt = input("\n💬 Enter a prompt (or 'quit' to exit): ").strip()
        
        if prompt.lower() in ['quit', 'exit']:
            print("👋 Goodbye!")
            break
        
        if not prompt:
            continue
        
        print(f"\n🔍 Responses to: '{prompt}'")
        print("-" * 40)
        
        for name, trainer in trainers.items():
            try:
                response = trainer.generate(prompt, max_length=30, temperature=0.7)
                # Extract just the generated part
                generated = response[len(prompt):].split('\n')[0][:50]
                if generated:
                    print(f"{name:>10}: {generated}")
                else:
                    print(f"{name:>10}: [generating...]")
            except Exception as e:
                print(f"{name:>10}: [error: {str(e)[:30]}]")

if __name__ == "__main__":
    main()

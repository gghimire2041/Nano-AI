"""
Flask API Server to serve all 5 AI models
==========================================

This creates a web API that serves all 5 AI models by importing from nano_AI.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch

# Import all models from your nano_AI.py file
from nano_AI import (
    SimpleTokenizer, 
    MiniChatGPT, 
    MiniGrok, 
    MiniGemini, 
    MiniDeepSeek, 
    MiniClaude,
    UniversalTrainer
)

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from your HTML

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
print("🤖 Initializing all 5 AI Models...")
tokenizer = SimpleTokenizer()
training_data = create_training_data()

# Fit tokenizer on all data
all_text = ' '.join(training_data.values())
tokenizer.fit(all_text)

# Initialize ALL 5 models
models = {
    'chatgpt': MiniChatGPT(tokenizer.vocab_size, d_model=32, n_heads=2, n_layers=2, max_seq_len=32),
    'grok': MiniGrok(tokenizer.vocab_size, d_model=32, n_heads=2, n_layers=2, max_seq_len=32, num_experts=2),
    'gemini': MiniGemini(tokenizer.vocab_size, d_model=32, n_heads=2, n_layers=2, max_seq_len=32),
    'deepseek': MiniDeepSeek(tokenizer.vocab_size, d_model=32, n_heads=2, n_layers=2, max_seq_len=32, num_experts=2),
    'claude': MiniClaude(tokenizer.vocab_size, d_model=32, n_heads=2, n_layers=2, max_seq_len=32)
}

trainers = {}

# Train each model
for name, model in models.items():
    print(f"🚀 Training {name}...")
    trainer = UniversalTrainer(model, tokenizer, name)
    trainer.train(training_data[name], epochs=15, lr=0.002)
    trainers[name] = trainer
    print(f"✅ {name} ready!")

print("🎉 All 5 models trained and ready!")

@app.route('/chat', methods=['POST'])
def chat():
    """API endpoint that returns REAL AI model responses from all 5 models"""
    try:
        data = request.json
        model_type = data.get('model', 'chatgpt')
        message = data.get('message', 'hello')
        
        if model_type not in trainers:
            available_models = list(trainers.keys())
            return jsonify({
                'error': f'Model {model_type} not available',
                'available_models': available_models
            }), 400
        
        # Get REAL neural network response
        trainer = trainers[model_type]
        real_response = trainer.generate(message, max_length=25, temperature=0.7)
        
        # Extract just the generated part (after the prompt)
        if message.lower() in real_response.lower():
            prompt_end = real_response.lower().find(message.lower()) + len(message)
            generated_part = real_response[prompt_end:].strip()
        else:
            generated_part = real_response
        
        # Clean up the response
        if not generated_part:
            generated_part = f"[{model_type} thinking...]"
        
        return jsonify({
            'response': generated_part,
            'model': model_type,
            'full_output': real_response,
            'status': 'success'
        })
        
    except Exception as e:
        print(f"❌ Error in /chat: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'OK', 
        'models': list(trainers.keys()),
        'total_models': len(trainers),
        'message': 'All AI models are running!'
    })

@app.route('/', methods=['GET'])
def home():
    """Root endpoint"""
    return jsonify({
        'message': 'AI Models API is running!',
        'available_endpoints': ['/health', '/chat'],
        'models': list(trainers.keys()) if trainers else []
    })

if __name__ == '__main__':
    print("🚀 Starting Flask API server with all 5 AI models...")
    print(f"💡 Models available: {list(models.keys())}")
    print("🌐 API endpoints:")
    print("  - GET  / (info)")
    print("  - GET  /health (health check)")
    print("  - POST /chat (chat with models)")
    app.run(debug=True, host='0.0.0.0', port=5000)

from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import torch
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

class IbexAI:
    def __init__(self):
        print("Initializing BlenderBot model...")
        self.tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
        self.model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.prompts = {
            "security": "You are IBEX, a witty cybersecurity AI companion. Answer with expertise and humor: ",
            "startup": "You are IBEX, a tech startup advisor AI. Respond with practical guidance: ",
            "deepai": "You are IBEX, an advanced AI researcher. Provide technical insights: ",
            "poetry": "You are IBEX, a creative AI storyteller. Compose thoughtfully: ",
            "general": "You are IBEX, a friendly AI security companion. Respond helpfully: "
        }

    def generate_response(self, message: str, intent: str = None) -> str:
        if not message:
            return "Please enter a message."

        # Default to general if no intent
        if not intent:
            intent = "general"

        prefix = self.prompts.get(intent, self.prompts["general"])
        prompt = prefix + message

        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        try:
            with torch.no_grad():
                reply_ids = self.model.generate(
                    inputs["input_ids"],
                    max_length=inputs["input_ids"].shape[1] + 50,
                    min_length=inputs["input_ids"].shape[1] + 10,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2
                )
            response = self.tokenizer.decode(reply_ids[0], skip_special_tokens=True)
            
            # Clean response - remove the prompt
            if prompt in response:
                response = response.replace(prompt, "").strip()
                
        except Exception as gen_err:
            print(f"[!] Model generation failed: {gen_err}")
            response = self.get_fallback_response(intent, message)

        return response

    def get_fallback_response(self, intent, message):
        fallback_templates = {
            "security": f"🛡️ I'm IBEX, your security AI! For '{message}' - always use strong passwords, enable 2FA, and stay vigilant!",
            "startup": f"🚀 Startup advice: solve real problems and iterate fast. About '{message}' - research your audience deeply!",
            "deepai": f"🤖 AI insight: focus on data quality and model architecture. Regarding '{message}' - deployment at scale matters!",
            "poetry": f"✨ A poem for you:\n\n'{message}' - like dawn breaking through code,\nBringing light to digital roads.",
            "general": f"Hey! I'm IBEX, your AI companion. About '{message}' - I'm here to help with anything you need!"
        }
        return fallback_templates.get(intent, f"I'm IBEX! I'm having trouble with that, but you said: '{message}'")

ibex_ai = IbexAI()

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        'service': 'IBEX AI Backend',
        'status': 'active',
        'version': '1.0.0',
        'description': 'Smart AI security companion with BlenderBot',
        'endpoints': ['/chat', '/api/ask', '/health']
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model': 'blenderbot-400M-distill'
    })

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Message required'}), 400
        
        message = data['message']
        context = data.get('context', 'general')
        
        response = ibex_ai.generate_response(message, context)
        
        return jsonify({
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'status': 'success',
            'model': 'blenderbot'
        })
    
    except Exception as e:
        return jsonify({
            'error': 'Processing failed',
            'message': str(e),
            'fallback_response': "I'm IBEX, your AI security companion! How can I help you today?"
        }), 500

@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json()
    message = data.get("message", "")
    intent = data.get("intent", "general")

    response = ibex_ai.generate_response(message, intent)
    return jsonify({"response": response})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

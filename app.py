from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import torch

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
            "security": "You are a cybersecurity expert. Answer the following question based on best security practices: ",
            "startup": "You are a tech startup advisor. Respond to the following inquiry with practical guidance: ",
            "deepai": "You are an advanced AI researcher. Provide deep technical insights into: ",
            "poetry": "You are a poetic storyteller AI. Compose a creative, thoughtful response to: ",
        }

    def generate_response(self, message: str, intent: str = None) -> str:
        if not message:
            return "Please enter a message."

        prefix = self.prompts.get(intent, "")
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
        except Exception as gen_err:
            print(f"[!] Model generation failed: {gen_err}")
            response = self.get_fallback_response(intent, message)

        return response

    def get_fallback_response(self, intent, message):
        fallback_templates = {
            "security": f"For cybersecurity, a good start is to regularly update software, use strong passwords, and enable 2FA. You asked: '{message}'",
            "startup": f"Startups thrive on solving real problems. Research your audience deeply and iterate fast. You asked: '{message}'",
            "deepai": f"AI development involves data quality, model architecture, and deployment at scale. You asked: '{message}'",
            "poetry": f"A poem begins in emotion and ends in understanding. Yours might start like this:\n\n\"{message}, the spark of dawn within the code.\"",
        }
        return fallback_templates.get(intent, f"I'm having trouble answering that. You said: '{message}'")

ibex_ai = IbexAI()

@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json()
    message = data.get("message", "")
    intent = data.get("intent", None)

    response = ibex_ai.generate_response(message, intent)
    return jsonify({"response": response})

@app.route("/", methods=["GET"])
def home():
    return "🧠 Ibex AI backend running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

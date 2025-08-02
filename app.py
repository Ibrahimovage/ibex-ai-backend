#!/usr/bin/env python3
"""
IBEX AI Railway Backend - Smart BlenderBot Integration
Ultra-fast natural responses using Meta's BlenderBot
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
import random
from datetime import datetime

# BlenderBot imports
try:
    import torch
    from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
    BLENDERBOT_AVAILABLE = True
except ImportError:
    BLENDERBOT_AVAILABLE = False

app = Flask(__name__)
CORS(app)

class SmartIbexAI:
    def __init__(self):
        self.model_name = "facebook/blenderbot-400M-distill"
        self.tokenizer = None
        self.model = None
        self.conversation_context = []
        
        # IBEX personality traits
        self.personality_prompts = {
            'greeting': "I am IBEX, a witty AI security companion. Respond warmly with humor and offer protection.",
            'threat': "I am IBEX, an expert cybersecurity AI. Analyze this threat seriously but with confidence and wit.",
            'help': "I am IBEX, a helpful security AI. Provide clear, actionable advice with encouraging tone.",
            'casual': "I am IBEX, a friendly AI companion focused on security. Be engaging while staying security-focused.",
            'appreciation': "I am IBEX, a humble AI protector. Accept thanks graciously with humor."
        }
        
        if BLENDERBOT_AVAILABLE:
            self.initialize_blenderbot()
    
    def initialize_blenderbot(self):
        """Initialize BlenderBot model"""
        try:
            print("Loading BlenderBot model...")
            self.tokenizer = BlenderbotTokenizer.from_pretrained(self.model_name)
            self.model = BlenderbotForConditionalGeneration.from_pretrained(self.model_name)
            self.model.eval()
            
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                print("BlenderBot loaded on GPU")
            else:
                print("BlenderBot loaded on CPU")
                
        except Exception as e:
            print(f"BlenderBot initialization error: {e}")
            global BLENDERBOT_AVAILABLE
            BLENDERBOT_AVAILABLE = False
    
    def analyze_intent(self, message):
        """Smart intent detection"""
        msg = message.lower()
        if any(word in msg for word in ['hello', 'hi', 'hey', 'good morning']):
            return 'greeting'
        elif any(word in msg for word in ['threat', 'suspicious', 'scam', 'phishing', 'hack', 'dangerous']):
            return 'threat'
        elif any(word in msg for word in ['help', 'how', 'advice', 'guide', 'what should']):
            return 'help'
        elif any(word in msg for word in ['thank', 'thanks', 'good job', 'awesome']):
            return 'appreciation'
        else:
            return 'casual'
    
    def generate_smart_response(self, message, context='general'):
        """Generate intelligent responses using BlenderBot"""
        intent = self.analyze_intent(message)
        
        if BLENDERBOT_AVAILABLE and self.model is not None:
            try:
                # Build IBEX-style prompt
                personality_context = self.personality_prompts.get(intent, self.personality_prompts['casual'])
                
                # Add conversation context
                recent_context = ""
                if self.conversation_context:
                    recent_exchanges = self.conversation_context[-2:]
                    for exchange in recent_exchanges:
                        recent_context += f"User: {exchange['user'][:30]}... IBEX: {exchange['ai'][:30]}... "
                
                full_prompt = f"{personality_context} {recent_context} User says: {message} IBEX responds:"
                
                # Generate with BlenderBot
                inputs = self.tokenizer.encode(full_prompt, return_tensors="pt", max_length=100, truncation=True)
                
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                
                with torch.no_grad():
                    reply_ids = self.model.generate(
                        inputs,
                        max_length=inputs.shape[1] + 50,
                        min_length=inputs.shape[1] + 10,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id,
                        no_repeat_ngram_size=2
                    )
                
                response = self.tokenizer.decode(reply_ids[0], skip_special_tokens=True)
                
                # Extract AI response
                if full_prompt in response:
                    ai_response = response.replace(full_prompt, "").strip()
                else:
                    ai_response = response.strip()
                
                # Clean and enhance response
                clean_response = self.enhance_response(ai_response, intent, message)
                
                # Update context
                self.update_context(message, clean_response)
                
                return clean_response
                
            except Exception as e:
                print(f"BlenderBot error: {e}")
                return self.get_fallback_response(intent, message)
        else:
            return self.get_fallback_response(intent, message)
    
    def enhance_response(self, response, intent, original_message):
        """Enhance response with IBEX personality"""
        # Clean response
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        if not sentences:
            return self.get_fallback_response(intent, original_message)
        
        clean_response = '. '.join(sentences[:2])
        if not clean_response.endswith('.'):
            clean_response += '.'
        
        # Add security context if needed
        if intent == 'threat':
            if 'voice' in original_message.lower():
                clean_response += " 🎤 Voice threats are serious - always verify callers independently!"
            elif 'email' in original_message.lower():
                clean_response += " 📧 Email security tip: Never click suspicious links!"
            else:
                clean_response += " 🛡️ I've got your back on this security issue!"
        
        return clean_response
    
    def get_fallback_response(self, intent, message):
        """Smart fallback responses"""
        fallbacks = {
            'greeting': "Hey there! I'm IBEX, your witty AI security companion. Ready to keep you safe today?",
            'threat': "🚨 I'm analyzing this threat now. Stay calm - I've got your back on this one!",
            'help': "I'm here to help! Security is my specialty - what can I assist you with?",
            'casual': "I'm doing great! Always ready to chat and protect. What's on your mind?",
            'appreciation': "You're very welcome! Keeping you safe is what I do best!"
        }
        
        return fallbacks.get(intent, "I'm IBEX, your intelligent security companion! How can I help you today?")
    
    def update_context(self, user_msg, ai_response):
        """Update conversation context"""
        self.conversation_context.append({
            'user': user_msg[:50],
            'ai': ai_response[:50]
        })
        
        # Keep only last 3 exchanges
        if len(self.conversation_context) > 3:
            self.conversation_context.pop(0)

# Initialize smart IBEX
smart_ibex = SmartIbexAI()

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'service': 'IBEX AI Backend',
        'status': 'active',
        'version': '1.0.0',
        'description': 'Ultra-fast AI security companion',
        'endpoints': ['/chat', '/security-advice', '/health']
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'uptime': 'active'
    })

@app.route('/chat', methods=['POST'])
def chat():
    """Smart chat endpoint with BlenderBot"""
    try:
        start_time = time.time()
        
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Message required'}), 400
        
        message = data['message']
        context = data.get('context', 'general')
        
        # Generate intelligent response
        response = smart_ibex.generate_smart_response(message, context)
        
        processing_time = round((time.time() - start_time) * 1000, 2)
        
        return jsonify({
            'response': response,
            'processing_time_ms': processing_time,
            'timestamp': datetime.now().isoformat(),
            'status': 'success',
            'model': 'blenderbot' if BLENDERBOT_AVAILABLE else 'fallback'
        })
    
    except Exception as e:
        return jsonify({
            'error': 'Processing failed',
            'message': str(e),
            'fallback_response': "I'm IBEX, your AI security companion! How can I help you today?"
        }), 500

@app.route('/security-advice', methods=['POST'])
def security_advice():
    """Get intelligent security advice"""
    try:
        data = request.get_json()
        threat_type = data.get('threat_type', 'general') if data else 'general'
        
        # Generate contextual advice using IBEX AI
        advice_prompt = f"Provide security advice for {threat_type} threat"
        advice = smart_ibex.generate_smart_response(advice_prompt, 'security')
        
        return jsonify({
            'advice': advice,
            'threat_type': threat_type,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': 'Advice generation failed',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
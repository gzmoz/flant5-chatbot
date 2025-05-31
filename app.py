from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "✅ API is running. Try POST /chat"})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("text", "")
    prompt = f"Answer this: {user_input}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=150)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": result})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

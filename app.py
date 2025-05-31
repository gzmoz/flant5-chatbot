
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

@app.route("/", methods=["GET"])
def home():
    return "FLAN-T5 Small API is running!"

@app.route("/chat", methods=["POST"])
def chat():
    try:
        input_text = request.json.get("text")
        if not input_text:
            return jsonify({"error": "Missing 'text' in request body"}), 400

        inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
        outputs = model.generate(**inputs, max_new_tokens=100)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({"response": decoded})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))

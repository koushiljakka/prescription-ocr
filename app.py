from flask import Flask, request, jsonify
from PIL import Image
import pytesseract
from difflib import get_close_matches

app = Flask(__name__)

known_medicines = ["Paracetamol", "Amoxicillin", "Ibuprofen", "Metformin", "Azithromycin", "Cetirizine"]

def check_medicines(text):
    words = text.split()
    results = []
    for word in words:
        match = get_close_matches(word, known_medicines, n=1, cutoff=0.6)
        if match:
            results.append({"extracted": word, "matched_medicine": match[0]})
    return results

@app.route("/", methods=["GET"])
def home():
    return "Smart Prescription Verification System is running."

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    image = Image.open(file)
    text = pytesseract.image_to_string(image)
    results = check_medicines(text)
    return jsonify({"extracted_text": text, "matches": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
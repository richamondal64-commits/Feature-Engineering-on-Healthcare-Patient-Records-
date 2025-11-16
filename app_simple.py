from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    return jsonify({"readmission_prediction": 1, "message": "Test response"})

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=3000)

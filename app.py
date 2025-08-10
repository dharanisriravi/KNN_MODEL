# app.py
from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__, static_folder="static", template_folder="templates")

# Load models
MODEL_PATH = "models/knn_model.pkl"
SCALER_PATH = "models/scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Please run train_model.py first to create models/knn_model.pkl and models/scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def hex_to_rgb(hex_color: str):
    """Convert #RRGGBB to tuple of ints (r,g,b)"""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        raise ValueError("Invalid color format")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return r, g, b

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    try:
        color = request.form.get("color", "#ffffff")
        size_cm = float(request.form.get("size_cm", 5.0))

        r, g, b = hex_to_rgb(color)
        features = np.array([[r, g, b, size_cm]])
        features_scaled = scaler.transform(features)
        pred = model.predict(features_scaled)[0]
        probs = model.predict_proba(features_scaled)[0]
        # Build class->prob mapping sorted descending
        classes = model.classes_
        class_probs = sorted(zip(classes, probs), key=lambda x: -x[1])

        return render_template("result.html", prediction=pred, class_probs=class_probs, color=color, size=size_cm)
    except Exception as e:
        return f"Error: {e}", 400

if __name__ == "__main__":
    app.run(debug=True)

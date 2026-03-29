import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

# ── Setup ──────────────────────────────────────────────────
app = Flask(__name__)

# Class names must match your dataset folder names alphabetically
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Bin color for each category (shown in the result)
BIN_INFO = {
    'cardboard': {'bin': 'Paper/Cardboard bin',  'color': '#4A90D9', 'emoji': '📦'},
    'glass':     {'bin': 'Glass bin',             'color': '#47B57A', 'emoji': '🍶'},
    'metal':     {'bin': 'Metal/Recycling bin',   'color': '#E8A838', 'emoji': '🥫'},
    'paper':     {'bin': 'Paper/Cardboard bin',   'color': '#4A90D9', 'emoji': '📄'},
    'plastic':   {'bin': 'Plastic bin',           'color': '#E85D5D', 'emoji': '♻️'},
    'trash':     {'bin': 'General waste bin',     'color': '#888888', 'emoji': '🗑️'},
}

# Load the trained model once when the app starts
print("Loading model...")
model = tf.keras.models.load_model('model/classifier.h5')
print("Model loaded successfully!")

# ── Helper function ────────────────────────────────────────
def predict_image(image_bytes):
    # Open image and resize to 224x224 (what the model expects)
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))

    # Convert to numpy array and normalize to 0-1
    img_array = np.array(img) / 255.0

    # Add batch dimension: (224, 224, 3) → (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # Run prediction
    predictions = model.predict(img_array, verbose=0)
    confidence = float(np.max(predictions))
    class_index = np.argmax(predictions)
    class_name = CLASS_NAMES[class_index]

    return class_name, confidence

# ── Routes ─────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    image_bytes = file.read()
    class_name, confidence = predict_image(image_bytes)

    # If confidence is below 60%, warn the user
    if confidence < 0.60:
        warning = "Low confidence — please double check manually"
    else:
        warning = None

    return jsonify({
        'class': class_name,
        'confidence': round(confidence * 100, 1),
        'bin': BIN_INFO[class_name]['bin'],
        'color': BIN_INFO[class_name]['color'],
        'emoji': BIN_INFO[class_name]['emoji'],
        'warning': warning
    })

if __name__ == '__main__':
    app.run(debug=True)
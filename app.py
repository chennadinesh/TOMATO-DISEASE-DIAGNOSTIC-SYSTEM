import os
import io
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.utils import load_img, img_to_array


app = Flask(__name__)

# ------------------ MODEL PATH (SAFE) ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "tomato_inception_v3.keras")

# ------------------ LOAD MODEL ------------------
MODEL = tf.keras.models.load_model(MODEL_PATH)

# ------------------ CLASS LABELS ------------------
CLASS_NAMES = [
    'Bacterial_Spot',
    'Early_Blight',
    'Healthy',
    'Late_Blight',
    'Septoria_Leaf_Spot',
    'Yellow_Leaf_Curl'
]

# ------------------ TEST TIME AUGMENTATION ------------------
def get_tta_predictions(img_array):
    """
    Input: img_array shape (1, 299, 299, 3)
    Output: Averaged prediction from TTA
    """

    img1 = img_array
    img2 = np.flip(img_array, axis=2)      # Horizontal flip
    img3 = np.flip(img_array, axis=1)      # Vertical flip
    img4 = np.rot90(img_array, k=1, axes=(1, 2))  # 90° rotation

    batch = np.vstack([img1, img2, img3, img4])

    predictions = MODEL.predict(batch)
    return np.mean(predictions, axis=0)


# ------------------ HOME ROUTE ------------------
@app.route('/')
def home():
    return render_template('index.html')


# ------------------ PREDICT ROUTE ------------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']

    # Read image directly from memory
    img = load_img(io.BytesIO(file.read()), target_size=(299, 299))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    averaged_probs = get_tta_predictions(img_array)

    predicted_index = np.argmax(averaged_probs)
    confidence = float(averaged_probs[predicted_index] * 100)
    disease_name = CLASS_NAMES[predicted_index]

    # Low confidence handling
    if confidence < 70:
        return jsonify({
            "disease": "Unclear_Input",
            "confidence": round(confidence, 2),
            "message": "Low confidence. Please upload a clearer tomato image."
        })

    return jsonify({
        "disease": disease_name,
        "confidence": round(confidence, 2)
    })


# ------------------ MEDICINE PAGE ------------------
@app.route('/medicine')
def medicine():
    disease = request.args.get('disease', 'Healthy')
    return render_template('medicine.html', disease=disease)


# ------------------ RUN SERVER ------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

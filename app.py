import os
import io
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.utils import load_img, img_to_array

<<<<<<< HEAD
app = Flask(__name__)

# ------------------ BASE DIR ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------ MODEL PATHS ------------------
LEAF_MODEL_PATH = os.path.join(BASE_DIR, "tomato_leaf_disease_model.keras")
FRUIT_MODEL_PATH = os.path.join(BASE_DIR, "tomato_fruit_disease_model.keras")

# ------------------ LOAD MODELS ------------------
LEAF_MODEL = tf.keras.models.load_model(LEAF_MODEL_PATH)
FRUIT_MODEL = tf.keras.models.load_model(FRUIT_MODEL_PATH)

# ------------------ CLASS LABELS ------------------
LEAF_CLASSES = [
    "Bacterial_spot",
    "Early_Blight",
    "Late_Blight",
    "Septoria_Leaf_Spot",
    "Yellow_Leaf_Curl_Virus",
    "healthy"
]

FRUIT_CLASSES = [
    "Anthracnose",
    "Bacterial_spot",
    "Early_Blight",
    "Late_Blight",
    "Fruit_Rot",
    "healthy"
]

IMG_SIZE = (224, 224)

# ------------------ TTA FUNCTION ------------------
def get_tta_predictions(model, img_array):

    img1 = img_array
    img2 = np.flip(img_array, axis=2)
    img3 = np.flip(img_array, axis=1)
    img4 = np.rot90(img_array, k=1, axes=(1, 2))

    batch = np.vstack([img1, img2, img3, img4])

    preds = model.predict(batch, verbose=0)
    return np.mean(preds, axis=0)


# ------------------ HOME ------------------
@app.route("/")
def home():
    return render_template("index.html")


# ------------------ PREDICT ------------------
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    plant_type = request.form.get("type", "leaf")
    file = request.files["file"]

    img = load_img(io.BytesIO(file.read()), target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Select model
    if plant_type == "fruit":
        model = FRUIT_MODEL
        classes = FRUIT_CLASSES
    else:
        model = LEAF_MODEL
        classes = LEAF_CLASSES

    averaged_probs = get_tta_predictions(model, img_array)

    idx = np.argmax(averaged_probs)
    confidence = float(averaged_probs[idx] * 100)
    disease = classes[idx]

=======

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
>>>>>>> 21c2ef7f91ac61ad54c70e517d0ceea8153757b3
    if confidence < 70:
        return jsonify({
            "disease": "Unclear_Input",
            "confidence": round(confidence, 2),
<<<<<<< HEAD
            "message": "Low confidence. Upload a clearer image."
        })

    return jsonify({
        "disease": disease,
        "confidence": round(confidence, 2),
        "type": plant_type
    })


# ------------------ MEDICINE ------------------
@app.route("/medicine")
def medicine():
    disease = request.args.get("disease", "healthy")
    return render_template("medicine.html", disease=disease)


# ------------------ RUN ------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
=======
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
>>>>>>> 21c2ef7f91ac61ad54c70e517d0ceea8153757b3

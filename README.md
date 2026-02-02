# 🌿 Tomato Disease Detection using Deep Learning

This project is an AI-based application that detects **tomato leaf diseases** and classifies them into different categories using image processing and deep learning. Farmers and researchers can use this tool to identify plant diseases early and take proper treatment actions.

---

## 📌 Features
- ✔ Detects multiple tomato leaf diseases
- ✔ Uses CNN / MobileNet / ResNet / InceptionV3 models
- ✔ Fast and accurate prediction
- ✔ Simple user interface (Flask Web App)
- ✔ Supports image upload
- ✔ Helps farmers reduce crop loss

---

## 🦠 Diseases Detected
- Early Blight  
- Late Blight  
- Septoria Leaf Spot  
- Bacterial Spot  
- Tomato Yellow Leaf Curl Virus   
- **Healthy Leaf**

---

## 🏗️ Project Architecture
1️⃣ Image Input (Upload Leaf Image)  
2️⃣ Preprocessing (Resize, Normalize)  
3️⃣ Model Prediction (CNN / MobileNet / ResNet / InceptionV3)  
4️⃣ Output Disease Name + Confidence  
5️⃣ (Optional) Suggest Possible Treatment

---

## 📂 Dataset
Dataset contains:
- Training images
- Validation images
- Healthy + Disease classes

Source:
- Plant Village Dataset
- Manually collected dataset

---

## 🛠️ Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Flask (for web app)

---

## 📦 Requirements
Install dependencies using:
```bash
pip install -r requirements.txt
```

---

## 🖥️ How to Run
### 1️⃣ Run Flask App
```bash
python app.py
```

### 2️⃣ Open Browser
```
http://127.0.0.1:5000/
```

Upload a tomato leaf image → Get result 🎯

---

## 📊 Model Details
This project uses multiple deep learning models:

- ✔ Custom CNN Model  
- ✔ MobileNetV2 Model  
- ✔ ResNet Model  
- ✔ **InceptionV3 Model** (Saved as `tomato_inception_v3.keras`)

To load model:
```python
from tensorflow.keras.models import load_model
model = load_model("tomato_inception_v3.keras")
```

---

## 📷 Output Example
✔ Upload leaf → AI Predicts Disease → Shows Result

---

## 🎯 Applications
- Farmers
- Agriculture researchers
- Smart farming systems
- Disease monitoring

---

## 🤝 Contributions
Feel free to modify or improve the project. Pull requests are welcome!

---

## 👨‍💻 Developed By
**C Chennadinesh**  
Mohan Babu University, Tirupati

---

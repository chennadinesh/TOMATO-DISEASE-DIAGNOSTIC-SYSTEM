🌿 Tomato Disease Detection System using Deep Learning & AI

This project is an AI-powered web application that detects tomato leaf and fruit diseases using deep learning models and image processing.

It helps farmers, students, and researchers identify plant diseases early and take proper treatment actions.

📌 Features

✔ Detects leaf and fruit diseases
✔ Supports CNN / MobileNet / ResNet / InceptionV3 / YOLO (Optional)
✔ Real-time image upload with drag & drop
✔ Background video UI
✔ Upload progress indicator
✔ Disease confidence score
✔ Medicine & prevention page
✔ Flask-based web application
✔ Simple & user-friendly interface
✔ Works on desktop & mobile

🦠 Diseases Detected
🌿 Leaf Diseases

Early Blight

Late Blight

Septoria Leaf Spot

Bacterial Spot

Tomato Yellow Leaf Curl Virus

Healthy Leaf

🍅 Fruit Diseases

Anthracnose

Fruit Rot

Healthy Fruit

🏗️ Project Architecture
User Upload Image
        ↓
Image Preprocessing
(Resize, Normalize, Augment)
        ↓
Deep Learning Model
(CNN / InceptionV3 / etc.)
        ↓
Prediction + Confidence
        ↓
Medicine Recommendation

📂 Dataset

Dataset Structure:

dataset/
 ├─ train/
 │   ├─ Early_Blight/
 │   ├─ Late_Blight/
 │   ├─ Healthy/
 │   └─ ...
 ├─ val/
 └─ test/

Sources

✔ PlantVillage Dataset
✔ Manually Collected Images
✔ Field Images

🛠️ Technologies Used
Category	Tools
Language	Python
AI/ML	TensorFlow, Keras
Image Processing	OpenCV, NumPy
Backend	Flask
Frontend	HTML, CSS, JavaScript
UI	Drag-Drop, Loader, Video BG
Version Control	Git, GitHub
📦 Requirements

Install dependencies:

pip install -r requirements.txt


Example requirements.txt:

tensorflow
flask
numpy
opencv-python
pillow

🖥️ How to Run the Project
1️⃣ Clone Repository
git clone https://github.com/yourusername/tomato-disease-detector.git
cd tomato-disease-detector

2️⃣ Run Flask Server
python app.py

3️⃣ Open Browser
http://127.0.0.1:5000/


✔ Upload Image
✔ Click Analyze
✔ View Result
✔ Check Medicine

📊 Model Details

This project uses multiple deep learning models:

✔ Custom CNN
✔ MobileNetV2
✔ ResNet
✔ InceptionV3 (Main Model)

Example: Load Model
from tensorflow.keras.models import load_model

model = load_model("tomato_inception_v3.keras")


For leaf and fruit:

LEAF_MODEL = load_model("tomato_leaf_disease_model.keras")
FRUIT_MODEL = load_model("tomato_fruit_disease_model.keras")

🎨 User Interface Features

✔ Background video
✔ Drag & Drop upload
✔ Highlight animation
✔ Loading spinner
✔ Upload percentage
✔ Progress bar
✔ Auto redirect to medicine page

📷 Output Example
Image Uploaded
↓
Disease: Early Blight
Confidence: 92%
↓
Medicine & Prevention Tips

🎯 Applications

🌾 Farmers
📊 Agriculture Researchers
🤖 Smart Farming Systems
📱 Mobile AI Apps
🏫 Academic Projects

🚀 Future Enhancements

✔ YOLO-based real-time detection
✔ Live camera scanning
✔ Mobile App (Android)
✔ Cloud Deployment
✔ Multi-language Support
✔ SMS Alert System

🤝 Contributions

Contributions are welcome!

Steps:

Fork the repository

Create feature branch

Commit changes

Create Pull Request

👨‍💻 Developed By

C Chennadinesh
BCA Student
Mohan Babu University, Tirupati

📬 Contact

📧 Email: yourmail@gmail.com

🔗 GitHub: https://github.com/yourusername
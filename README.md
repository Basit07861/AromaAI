# 🌸 AromaAI 🌿

> Flower image classification powered by EfficientNet — future-ready to generate scents for a multisensory AI experience.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python" alt="Python Badge"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.13-orange?logo=tensorflow" alt="TensorFlow Badge"/>
  <img src="https://img.shields.io/badge/Keras-%23D00000?logo=keras&logoColor=white" alt="Keras Badge"/>
  <img src="https://img.shields.io/badge/EfficientNetB0-Transfer%20Learning-green" alt="EfficientNet Badge"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License Badge"/>
</p>

---

## ✨ Overview
**AromaAI** is a deep learning project that classifies flower images into multiple categories using transfer learning with EfficientNetB0.  
The project is designed as the first step towards building an **AI-powered scent generation device** that could recognize flowers and produce their corresponding aromas.

---

## 🧩 Features
✅ Built with TensorFlow and Keras  
✅ Uses EfficientNetB0 pretrained on ImageNet  
✅ Data augmentation & fine-tuning for better accuracy  
✅ Plots training curves for accuracy & loss  
✅ Modular code for easy extension  
✅ Future plan: integrate scent generation device

---

## 📦 Project Structure

AromaAI/   
├── data/                   
├── test_images/            
├── flower_classifier.py    
├── test_classifier.py      
├── flower_model.h5         
├── requirements.txt  
└── README.md


---

## ⚙ Setup & Installation

1️⃣ Clone the repository:

git clone https://github.com/Basit07861/AromaAI.git  
cd AromaAI

2️⃣ Create & Activate Virtual Environment:

python -m venv venv  
.\venv\Scripts\activate           # Windows

# or

source venv/bin/activate          # macOS/Linux

3️⃣ Install dependencies:

pip install -r requirements.txt

🚀 Usage
## 1. Train the model

Make sure your dataset is extracted inside the data/ folder.

python flower_classifier.py

This will:

Train the model

Fine-tune EfficientNetB0

Save the trained model as flower_model.h5

## 2. Test on a new image
   
Put your test image inside test_images/ and update the filename in test_classifier.py.

Then run:

python test_classifier.py  
You'll see the predicted flower class & probabilities.

🌱 Future work
Integrate model output with a hardware device (e.g., Raspberry Pi + scent module)

Generate real flower aromas based on detected class

Build a simple web app or mobile interface

✏️ License
This project is open-source and free to use under the MIT License.

🤝 Contributing
Feel free to fork, raise issues, or submit pull requests to improve AromaAI!

🚀 Built with curiosity & creativity, aiming to bring sight & scent together through AI.

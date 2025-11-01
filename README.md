# 👟 Nike vs Adidas Shoe Image Classifier — Deep Learning with MobileNetV2

This project is a **Streamlit web application** that classifies uploaded shoe images as either **Nike** or **Adidas** using a fine-tuned **MobileNetV2** deep learning model trained with TensorFlow/Keras.

---

## 🚀 Project Overview

The goal of this project is to build and deploy a lightweight **image classification** model that can distinguish between Nike and Adidas shoes based on visual features.

The app allows users to **upload an image**, view the **predicted brand**, and see the **confidence level** for each prediction.

---

## 🧠 Tech Stack

- **Python 3.11**
- **TensorFlow / Keras (MobileNetV2)**
- **Streamlit** (Web App Interface)
- **Pillow (PIL)** for image processing
- **NumPy, Matplotlib** for numerical and plotting utilities
- **Google Colab** for training
- **VS Code / Streamlit Cloud** for deployment

---

## 🧩 Project Structure
```
Nike_Adidas_App/
│
├── app.py              # Streamlit web app
├── mobilenetv2_shoes_final_finetuned.keras
                        # Trained model file (27 MB)
├── requirements.txt    # Dependencies
```
---

## 🧪 Model Details

- **Architecture:** MobileNetV2 (pre-trained on ImageNet)
- **Transfer Learning:** Frozen base layers + fine-tuned top layers
- **Optimizer:** Adam (learning rate = 1e-4)
- **Loss Function:** Categorical Crossentropy
- **Augmentation:** Random flips, rotations, zoom, and brightness shifts
- **Accuracy:** ~70%–79% (depending on dataset balance)

---

## ⚙️ Installation and Setup

### 1️⃣ Clone the Repository  

```bash
git clone https://github.com/yourusername/nike-adidas-classifier.git

cd nike-adidas-classifier
```
---

### 2️⃣ (Optional) Create Virtual Environment
```bash
python -m venv venv
venv\\Scripts\\activate  # On Windows
# or
source venv/bin/activate  # On macOS/Linux
```
---
### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
---

### 🧰 requirements.txt

streamlit
tensorflow==2.17.0
keras==3.3.3
protobuf==3.20.3
pillow
 
 ---
 
## 🎯 Usage
### ▶️ Run the App Locally

Then open your browser at:
http://localhost:8501

--- 
### 🖼️ Upload an Image

Click Browse files

Select any Nike or Adidas shoe image

The model will output:

🏷️ Predicted brand (Nike / Adidas)

🔢 Confidence level (%)
 
---

## 🌍 Deployment

### 🚢 Streamlit Cloud (Recommended)

1. Push this project to GitHub

2. Go to https://share.streamlit.io

3. Log in with GitHub

4. Click New App → Choose your repo → Select app.py

5. Click Deploy

You’ll get a shareable public URL like:
```bash
https://yourname-nike-adidas-classifier.streamlit.app
```
---

## 🧱 Model Download (If Too Large for GitHub)

If your .keras model exceeds 100 MB:

1. Upload it to Google Drive

2. Get a sharable link

3. Update app.py to download automatically via gdown

```bash
import gdown, os, tensorflow as tf

url = "https://drive.google.com/uc?id=YOUR_FILE_ID"
output = "mobilenetv2_shoes_final_finetuned.keras"

if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

model = tf.keras.models.load_model(output)
```
---

## 🧠 Future Improvements

* Improve dataset balance (more equal Nike vs Adidas images)

* Fine-tune more MobileNetV2 layers

* Add Grad-CAM visualization (to show where model focuses)

* Extend to multi-class (e.g., Puma, Reebok, etc.)

---

## 🏁 Author


👤 Arbi Rayen
🎓 Data Science & Deep Learning Enthusiast
💼 AI / ML Freelancer | Computer Vision Developer
📧 Contact: arbirayen123.@gmail.com

---

## ⭐ Acknowledgements

* TensorFlow Team for MobileNetV2

* Streamlit for making AI deployment simple

* Google Colab for model training environment

---

## 🏆 License

This project is released under the MIT License.
You are free to use, modify, and distribute with attributionn 

Streamlit for making AI deployment simple

Google Colab for model training environment

🏆 License

This project is released under the MIT License.
You are free to use, modify, and distribute with attribution.


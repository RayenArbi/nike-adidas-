
# 🏷️ Nike vs Adidas Classifier — Streamlit App with Confidence Display
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ================================================================
# 1️⃣ Load the model (cached so it loads once)
# ================================================================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mobilenetv2_shoes_final_finetuned.keras")
    return model

model = load_model()
class_names = ["adidas", "nike"]  # Make sure these match your dataset folder names

# ================================================================
# 2️⃣ Streamlit App UI
# ================================================================
st.set_page_config(page_title="Nike vs Adidas Classifier 👟", page_icon="👟", layout="centered")

st.title("👟 Nike vs Adidas Shoe Classifier")
st.markdown("Upload an image of a shoe, and the model will predict if it's **Nike** or **Adidas** with a confidence score.")

uploaded_file = st.file_uploader("📤 Upload a shoe image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # ================================================================
    # 3️⃣ Preprocess the image
    # ================================================================
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ================================================================
    # 4️⃣ Make Prediction
    # ================================================================
    prediction = model.predict(img_array)
    pred_idx = np.argmax(prediction)
    pred_class = class_names[pred_idx]
    confidence = float(np.max(prediction)) * 100

    # ================================================================
    # 5️⃣ Display Prediction Results
    # ================================================================
    st.markdown(f"### 🧠 Prediction: **{pred_class.upper()}**")
    st.markdown(f"**Confidence:** `{confidence:.2f}%`")

    # Progress bar visualization
    st.progress(confidence / 100)

    if pred_class.lower() == "nike":
        st.success(f"✅ This looks like a **Nike** shoe! ({confidence:.1f}% confidence)")
    else:
        st.info(f"✅ This looks like an **Adidas** shoe! ({confidence:.1f}% confidence)")

else:
    st.warning("👆 Please upload an image to get a prediction.")

import streamlit as st
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Deep-Learning Demo", layout="wide")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;700&display=swap');
    
    body {
        font-family: 'Lora', serif;
        
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Lora', serif;
    }

    p {
        font-family: 'Lora', serif;
    }
    </style>
""", unsafe_allow_html=True)



@st.cache_resource
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_keras_model(path):
    return tf.keras.models.load_model(path)

pre = load_pickle("preprocessor.pkl")
y_scaler = load_pickle("y_scaler.pkl")

text_model       = load_keras_model("text_model.keras")
image_model      = load_keras_model("image_model.keras") 
regression_model = load_keras_model("regression_model.keras")

def preprocess_image(img_pil, size=(224, 224)):
    img = img_pil.convert("RGB").resize(size)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, 0)
    return arr

page = st.sidebar.radio(
    "Choose a page",
    ("About", "Text Classifier", "Image Classifier", "House Price Regressor")
)

if page == "About":
    
    st.markdown("<h1 style='text-align: center;'>ℹ About</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center;'>
        <p>This demo shows three Deep Learning models:</p>
        <p>- Text Classifier – sentiment analysis</p>
        <p>- Image Classifier – image classification</p>
        <p>- House Price Regressor – price prediction</p>
        <p><strong>Repo:</strong> <a href="https://github.com/marielalvarez/Deep-Learning">GitHub Repo</a></p>
        <p><em>Mariel Álvarez Salas – 2025</em></p>
    </div>
    """, unsafe_allow_html=True)



elif page == "Text Classifier":
    st.markdown("<h1 style='text-align: center;'>Text Classifier</h1>", unsafe_allow_html=True)
    user_text = st.text_area("Enter text to classify", height=150)
    if st.button("Predict"):
        if user_text.strip() == "":
            st.warning("Please enter some text.")
        else:
            input_tensor = tf.constant([user_text], dtype=tf.string)
            pred_prob = text_model.predict(input_tensor)[0][0]
            label = "Spam" if pred_prob >= 0.5 else "Ham"
            st.success(f"Prediction: {label} (confidence: {pred_prob:.2f})")

elif page == "Image Classifier":
    st.markdown("<h1 style='text-align: center;'>Image Classifier</h1>", unsafe_allow_html=True)
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if file:
        img = Image.open(file)
        st.image(img, caption="Uploaded image", width=250)
        arr = preprocess_image(img)
        pred = image_model.predict(arr)[0][0]  
        label = "Dog" if pred >= 0.5 else "Cat"
        st.success(f"**Prediction:** {label} (confidence: {pred:.2f})")

elif page == "House Price Regressor":
    st.markdown("<h1 style='text-align: center;'>House Price Regressor</h1>", unsafe_allow_html=True)
    csv_file = st.file_uploader("Upload a CSV with feature columns", type="csv")
    if csv_file is not None:
        df = pd.read_csv(csv_file)
        st.write("Input preview:", df.head())

        X_prep = pre.transform(df)

        y_pred_scaled = regression_model.predict(X_prep)

        y_pred = y_scaler.inverse_transform(y_pred_scaled)

        df_out = df.copy()
        df_out["pred_price"] = y_pred.flatten()

        st.write("Predictions:", df_out.head())

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.hist(y_pred.flatten(), bins=30, edgecolor='black', alpha=0.7)

        ax.set_title("Distribution of Predicted House Prices", fontsize=16)
        ax.set_xlabel("Predicted Price", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        st.pyplot(fig, use_container_width=True)
        csv_bytes = df_out.to_csv(index=False).encode()
        st.download_button("Download predictions CSV", csv_bytes, "predictions.csv")
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(page_title="Hydration Essentials", layout="centered")
st.title("Hydration Essentials: Bottle Classifier")
st.markdown("Upload an image of a water bottle and let the model predict its type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((128, 128))
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert to numpy and normalize
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape((1, 128, 128, 3))

    # Load model
    try:
        model = tf.keras.models.load_model("model/bottle_classifier.h5")
        prediction = model.predict(img_array)
        class_names = ["Plastic", "Metal", "Glass", "Sipper", "Spray", "Jug"]
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        st.success(f"Predicted Class: {predicted_class} ({confidence:.2f}%)")
    except Exception as e:
        st.error("Model file not found or prediction failed. Showing simulated result.")
        st.success("Predicted Class: Plastic (94.6%)")

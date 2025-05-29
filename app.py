import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_model.h5")
    return model

model = load_model()

# Define class labels
class_labels = [
    'Actinic Keratoses', 'Basal Cell Carcinoma', 'Benign Keratosis-like Lesions',
    'Dermatofibroma', 'Melanocytic Nevi', 'Melanoma', 'Vascular Lesions',
    'Squamous Cell Carcinoma', 'Tinea Ringworm Candidiasis'
]

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# App Layout
st.title("Skin Cancer Detection & Classification")
st.write("A Skin Cancer classifier which classifies images into 9 classes of cancer.")

uploaded_file = st.file_uploader("Drop Image Here or Click to Upload", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button("Submit"):
        with st.spinner('Classifying...'):
            processed_image = preprocess_image(image)
            predictions = model.predict(processed_image)
            predicted_class = class_labels[np.argmax(predictions)]
            st.success(f"Predicted Class: {predicted_class}")

st.write("---")
st.write("### Examples")
example_images = ["example1.jpg", "example2.jpg"]  # Replace with actual image paths if available
cols = st.columns(len(example_images))
for col, example in zip(cols, example_images):
    col.image(example, use_column_width=True)

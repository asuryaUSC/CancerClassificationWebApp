import sys
print(sys.executable)

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Show a spinner while the model is being loaded
with st.spinner('Loading model...'):
    model = tf.keras.models.load_model('/Users/aadityasurya/Desktop/Cancer Research/Full Code/best_model.h5')

def predict_tumor(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    
    labels = ['Normal', 'Benign', 'Malignant']
    return labels[predicted_class]

# UI enhancements
st.markdown("<h1 style='text-align: center; color: black;'>Tumor Classification App</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: grey;'>Upload an image and classify whether a tumor is normal, benign, or malignant.</h3>", unsafe_allow_html=True)

st.write("")

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'])

if uploaded_file is not None:
    st.write("")
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Show a spinner while the image is being classified
    with st.spinner('Classifying...'):
        label = predict_tumor(image)

    st.markdown(f"<h3 style='text-align: center; color: red;'>This tumour is: {label}</h3>", unsafe_allow_html=True)

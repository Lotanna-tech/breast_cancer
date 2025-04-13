import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import tf_keras

# Define the URL of the feature extractor
feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

# Create the KerasLayer
feature_extractor_layer = hub.KerasLayer(feature_extractor_url, trainable=False, input_shape=(224, 224, 3))

# Define the model structure
model = tf_keras.Sequential([
    feature_extractor_layer,
    tf_keras.layers.Dense(3, activation='softmax')
])

# Now, load the weights from your saved file
model.load_weights('breast_classification_model.h5')

# Streamlit title
st.title("Breast Cancer Classification")

# Image upload widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Pre-process the image (resize and normalize)
    image = image.resize((224, 224))
    image = np.array(image)
    image = image / 255.0  # Normalize the image

    # Make prediction using the model
    prediction = model.predict(np.expand_dims(image, axis=0))
    prediction_label = np.argmax(prediction, axis=1)

    # Map prediction to labels
    labels = ["benign", "malignant", "normal"]
    result = labels[prediction_label[0]]

    # Display the prediction result
    st.write(f"Prediction: {result}")
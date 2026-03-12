import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image
import io
import os
from huggingface_hub import hf_hub_download

# Set page configuration
st.set_page_config(page_title="Face Mask Detection", layout="centered")

# Title and description
st.title("🔍 Face Mask Detection")
st.markdown("Upload an image to detect whether a face is wearing a mask or not")



# Load model (cached for efficiency)
@st.cache_resource
def load_model():
    # Download model from Hugging Face Hub
    model_path = hf_hub_download(
        repo_id="Recurrent/face_mask_dectection",  # Replace with your HF username
        filename="face_mask_model.h5",
        cache_dir="./model_cache"
    )
    model = tf.keras.models.load_model(model_path)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])


#-------------------------------------------------------------------------------------------------------

if uploaded_file is not None:
    try:
        # Read and display image
        image = Image.open(uploaded_file)
        
        # Convert image to RGB if needed (handles RGBA, grayscale, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_rgb = np.array(image)
        
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image_rgb, use_column_width=True)
        
        # Preprocess image
        img_resized = cv2.resize(image_rgb, (128, 128))
        img_array = np.array(img_resized, dtype='float32')
        
        # Normalize using the same layer as training
        normalization_layer = layers.Rescaling(1./255)
        img_normalized = normalization_layer(img_array)
        
        # Add batch dimension
        img_input = np.expand_dims(img_normalized, axis=0)
        
        # Make prediction
        y_pred = model.predict(img_input, verbose=0)
    
        # Display results
        with col2:
            st.subheader("Prediction Result")
            
            mask_confidence = y_pred[0][0]
            no_mask_confidence = y_pred[0][1]
            
            if mask_confidence > no_mask_confidence:
                st.success(f"✅ **Mask Detected**")
                st.metric("Confidence", f"{mask_confidence*100:.2f}%")
            else:
                st.warning(f"⚠️ **No Mask Detected**")
                st.metric("Confidence", f"{no_mask_confidence*100:.2f}%")
        
        # Show detailed probabilities
        st.divider()
        st.subheader("Detailed Predictions")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mask Probability", f"{mask_confidence*100:.2f}%")
        with col2:
            st.metric("No Mask Probability", f"{no_mask_confidence*100:.2f}%")
        
        # Probability chart
        probs = {
            "With Mask": mask_confidence,
            "Without Mask": no_mask_confidence
        }
        st.bar_chart(probs)
    
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        st.info("Please make sure the image is a valid image file (JPG, PNG, etc.)")
else:
    st.info("👆 Please upload an image to get started")

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: small;'>
    Face Mask Detection Model | ResNet50 Architecture
    </div>
    """,
    unsafe_allow_html=True
)

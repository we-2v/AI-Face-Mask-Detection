# --- Import Necessary Libraries ---
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import os

# Set Hugging Face Token for faster and authenticated downloads
os.environ["HF_TOKEN"] = "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"

# --- 1. CONFIGURATION & MODEL LOADING ---

# Set Streamlit page configuration for title, icon, and layout
st.set_page_config(
    page_title="AI Face Mask Detector",
    page_icon="😷",
    layout="centered", # 'centered' or 'wide'
    initial_sidebar_state="auto" # 'auto', 'expanded', or 'collapsed'
)

# Function to load the AI model from Hugging Face Hub
# @st.cache_resource ensures the model is downloaded and loaded only once across sessions
@st.cache_resource
@st.cache_resource
def load_trained_model():
    REPO_ID = "We-2V/face-mask-detection-resnet50"
    FILENAME = "face_mask_model_v5_augmented.keras" 

    try:
        with st.spinner('⏳ Downloading AI brain from Hugging Face... Please wait.'):
            model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

            base_model_inference = tf.keras.applications.ResNet50(
                input_shape=(128, 128, 3), 
                include_top=False, 
                weights='imagenet'
            )
            base_model_inference.trainable = False 

            final_inference_model = tf.keras.models.Sequential([
                tf.keras.layers.Input(shape=(128, 128, 3)),
                base_model_inference,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(2, activation='softmax')
            ])

            final_inference_model.load_weights(model_path)
            
            return final_inference_model
            
    except Exception as e:
        st.error(f"Critical Error loading model from Hugging Face: {e}")
        st.error("This often happens with models containing specific Keras layers (like Data Augmentation) saved as '.keras' files.")
        st.error("Possible solutions: 1. Ensure your Hugging Face repo and filename are correct.")
        st.error("                 2. Try uploading a model saved WITHOUT Data Augmentation layers (e.g., your v3.h5 model if it exists).")
        return None


# --- 2. IMAGE PROCESSING & PREDICTION LOGIC ---

# Function to preprocess the image to match the model's input requirements
def preprocess_image(image: Image.Image) -> np.ndarray:
    # Convert image to RGB format and resize to 128x128 pixels
    img = image.convert('RGB')
    img = img.resize((128, 128))
    # Convert to NumPy array and normalize pixel values from 0-255 to 0-1
    img_array = np.array(img, dtype='float32') / 255.0
    # Add a batch dimension: (1, 128, 128, 3) for the model
    img_input = np.expand_dims(img_array, axis=0)
    return img_input

# Function to get prediction from the loaded model
def get_prediction(image: Image.Image, model: tf.keras.Model) -> tuple:
    processed_img = preprocess_image(image)
    # Get raw prediction probabilities from the model
    prediction_probabilities = model.predict(processed_img, verbose=0)[0]
    
    # Define labels (make sure this order matches your model's training)
    labels = ["With Mask", "Without Mask"] 
    
    # Get the index of the highest probability
    result_idx = np.argmax(prediction_probabilities)
    confidence = prediction_probabilities[result_idx] * 100
    
    return labels[result_idx], confidence, result_idx

# --- 3. USER INTERFACE (UI) LAYOUT ---

# Display the main title and introduction
st.title("😷 AI Face Mask Detection System")
st.write("An Artificial Intelligence application to detect face masks in real-time. Built by **Eng. Shehab Ameen**.")
st.markdown("---")

# Load the model; the rest of the UI only appears if the model loads successfully
model = load_trained_model()

if model: # Proceed only if the model is loaded
    st.success("✅ AI brain is ready and operational!")
    
    # User input method selection (Upload or Camera)
    input_option = st.radio(
        "Choose your input method:", 
        ("Upload an Image", "Use Live Camera"), 
        horizontal=True
    )
    
    img_file_buffer = None
    if input_option == "Upload an Image":
        img_file_buffer = st.file_uploader("Select an image from your device:", type=["jpg", "png", "jpeg"])
    else: # Use Live Camera
        img_file_buffer = st.camera_input("Center your face in the frame and take a picture")

    # Process and display results if an image is provided
    if img_file_buffer is not None:
        # Open the image using PIL
        image = Image.open(img_file_buffer)
        
        # Display the uploaded/captured image
        st.image(image, caption='Your Input Image', use_column_width=True)
        
        # Add a button to trigger the analysis
        if st.button('Analyze Image 🚀'):
            # Get prediction and confidence
            result_label, confidence, result_idx = get_prediction(image, model)
            
            st.markdown("---")
            if result_idx == 0: # Assuming 0 is 'With Mask'
                st.success(f"**RESULT:** {result_label} (Confidence: {confidence:.2f}%)")
            else: # Assuming 1 is 'Without Mask'
                st.error(f"**RESULT:** {result_label} (Confidence: {confidence:.2f}%)")
            st.markdown("---")

# Footer for the application
st.markdown("---")
st.write("Built with using TensorFlow, Keras, Streamlit, and Hugging Face.")
st.write("For more details, visit the [GitHub Repository](https://github.com/we-2v/deep_learning_projects)")

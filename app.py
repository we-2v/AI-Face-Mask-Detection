# ====================================================
# Face Mask Detection Web Application by Shehab Ameen
# ====================================================

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from huggingface_hub import hf_hub_download
import os

# --- 1. CONFIGURATION & MODEL LOADING ---

# إعداد شكل وعنوان الصفحة
st.set_page_config(
    page_title="AI Face Mask Detector",
    page_icon="😷",
    layout="centered",
    initial_sidebar_state="auto"
)

# دالة تحميل الموديل من Hugging Face (تعمل مرة واحدة فقط)
@st.cache_resource
def load_trained_model():
    # معلومات حسابك ومستودعك
    REPO_ID = "We-2V/face-mask-detection-resnet50"
    # اسم ملف الموديل الذي قمت برفعه
    FILENAME = "face_mask_model_v5_augmented.keras" # أو "face_mask_model_v3.h5"

    try:
        # عرض رسالة انتظار أنيقة أثناء التحميل
        with st.spinner('⏳ Downloading AI brain from Hugging Face... Please wait.'):
            model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
            # تحميل الموديل باستخدام Keras
            model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        # عرض رسالة خطأ واضحة للمستخدم
        st.error(f"❌ Error loading model: {e}")
        st.error("Please make sure the REPO_ID and FILENAME are correct in the app.py file.")
        return None

# --- 2. IMAGE PROCESSING & PREDICTION LOGIC ---

# دالة معالجة الصورة لتناسب النموذج
def preprocess_image(image):
    img = image.convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img, dtype='float32') / 255.0
    img_input = np.expand_dims(img_array, axis=0)
    return img_input

# --- 3. USER INTERFACE (UI) ---

# العنوان الرئيسي للتطبيق
st.title("😷 Face Mask Detection System")
st.write("An AI-powered application to detect face masks in real-time. Built by **Eng. Shehab Ameen**.")
st.markdown("---")

# تحميل الموديل وعرض رسالة نجاح
model = load_trained_model()

# فقط إذا تم تحميل الموديل بنجاح، نعرض باقي الواجهة
if model:
    st.success("✅ AI brain is ready and operational!")
    
    # خيارات للمستخدم: رفع صورة أو استخدام الكاميرا
    option = st.radio("Choose your input method:", ("Upload an Image", "Use Live Camera"), horizontal=True)
    
    img_file_buffer = None
    if option == "Upload an Image":
        img_file_buffer = st.file_uploader("Select an image from your device:", type=["jpg", "png", "jpeg"])
    else:
        img_file_buffer = st.camera_input("Center your face in the frame and take a picture")

    # إذا قام المستخدم برفع صورة أو التقاطها
    if img_file_buffer is not None:
        # قراءة الصورة وعرضها
        image = Image.open(img_file_buffer)
        st.image(image, caption='Your Image', use_column_width=True)

        # تجهيز الصورة للتوقع
        processed_img = preprocess_image(image)

        # التوقع
        prediction = model.predict(processed_img, verbose=0)[0]
        
        # ترتيب الفئات (0: كمامة, 1: بدون كمامة)
        labels = ["With Mask", "Without Mask"] 
        result_idx = np.argmax(prediction)
        confidence = prediction[result_idx] * 100
        
        # عرض النتيجة بألوان مختلفة
        st.markdown("---")
        if result_idx == 0:
            st.success(f"**RESULT:** {labels[0]} (Confidence: {confidence:.2f}%)")
        else:
            st.error(f"**RESULT:** {labels[1]} (Confidence: {confidence:.2f}%)")

st.markdown("---")
st.write("Powered by TensorFlow, Keras, and Streamlit. Deployed via Hugging Face.")

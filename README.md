# AI-Powered Face Mask Detection 🛡️🤖
### Enhanced with Deep Transfer Learning & Strategic Fine-Tuning

This repository features an optimized version of a Face Mask Detection system. The project has been significantly enhanced through rigorous architectural refactoring, advanced training methodologies, and deployment optimization, achieving a validation accuracy of **85%**.

---

## 🚀 Key Improvements
* **Performance Boost:** Increased validation accuracy to 85% through precision hyperparameter tuning.
* **Strategic Fine-Tuning:** Implemented a sophisticated two-stage training process.
* **Cloud Deployment:** The trained model is securely hosted on Hugging Face and deployed via Streamlit, making it accessible to everyone as an interactive web application.

---

## 🏗️ Project Pipelines

### 1. Model Building Pipeline
The diagram below illustrates the end-to-end process of building the model, from data preparation to model saving:
![Model Building Pipeline](https://ibb.co/pj1KknTH)

### 2. Deployment Pipelines
**Local Inference Architecture:**
![Local Deployment Pipeline](https://ibb.co/B5yNJ39t)

**Cloud Deployment Workflow:**
The model is hosted on Hugging Face, the codebase on GitHub, and the user interface is deployed via Streamlit Cloud:
![Cloud Deployment Pipeline](https://ibb.co/KxN3VqSz)

---

## 🛠️ Technical Stack
* **Deep Learning Framework:** TensorFlow & Keras
* **Computer Vision:** OpenCV (for real-time face detection and preprocessing)
* **Base Architecture:** ResNet50 (Pre-trained on ImageNet)
* **Model Hosting:** Hugging Face Hub 🤗
* **Web Deployment:** Streamlit Cloud 🌐
* **Data Strategy:** Advanced Data Augmentation (Rotation, Zoom, Horizontal Flip)

---

## 🧠 Methodology: Two-Stage Fine-Tuning
A transfer learning approach was adopted using **ResNet50** as the feature extractor, optimized with the **Adam** optimizer:

1. **Stage 1 (Feature Extraction):** The entire ResNet50 backbone was **frozen**. Only custom-added top layers were trained to map pre-trained features to specific classes (Mask vs. No Mask).
2. **Stage 2 (Fine-Tuning):** The **last 10 layers** of ResNet50 were **unfrozen** and re-trained with a lower learning rate. This specialized the model for facial features and mask textures, stabilizing the loss curve and boosting accuracy.

---

## 📊 Results & Performance
* **Validation Accuracy:** 85%
* **Optimizer:** Adam
* **Learning Stability:** Smooth convergence in the loss function post-fine-tuning, indicating effective weight adaptation without catastrophic forgetting.

---

## 📂 Project Structure
* `face_mask_1.ipynb`: Comprehensive Jupyter Notebook containing data preprocessing and the complete model training pipeline.
* `app.py`: Main Streamlit application script for the web interface.
* `requirements.txt`: Environment dependencies and required libraries.
* `models/`: Local directory for the trained model in `.keras` format (Model is also hosted on Hugging Face for deployment).

---

## 🔗 Resources & Links
* **Live Web App:** [Try it on Streamlit](https://ai-face-mask-detection1.streamlit.app/)
* **Demonstration Video:** [Watch on YouTube](https://youtu.be/E71ms9MDlOA)
* **Dataset:** [Download via Google Drive](https://drive.google.com/file/d/1a7Brk_hSxQPCAbPUF-qx0yPr-tlaljjK/view?usp=sharing)

# AI-Powered Face Mask Detection 🛡️🤖
### Enhanced with Deep Transfer Learning & Strategic Fine-Tuning

This repository features an optimized version of the Face Mask Detection system, originally forked from `recurrentai/deep_learning_projects`. The project has been significantly enhanced through rigorous architectural refactoring, advanced training methodologies, and deployment optimization, achieving a validation accuracy of **85%**.

---

## 🚀 Key Improvements in This Version
* **Performance Boost:** Increased validation accuracy to 85% through precision hyperparameter tuning.
* **Strategic Fine-Tuning:** Implemented a sophisticated two-stage training process.
* **Code Refactoring:** Optimized the codebase for better modularity, readability, and inference speed.
* **Deployment Ready:** Fully configured for seamless deployment on Streamlit Cloud with cross-platform dependency management.

---

## 🛠️ Technical Stack
* **Deep Learning Framework:** TensorFlow & Keras
* **Computer Vision:** OpenCV (for real-time face detection and preprocessing)
* **Base Architecture:** ResNet50 (Pre-trained on ImageNet)
* **Deployment:** Streamlit
* **Data Strategy:** Advanced Data Augmentation (Rotation, Zoom, Horizontal Flip) to enhance model generalization.

---

## 🧠 Methodology: The Two-Stage Fine-Tuning Process
To achieve optimal results, a transfer learning approach was adopted using **ResNet50** as the feature extractor, optimized with the **Adam** optimizer:

### Stage 1: Feature Extraction
The entire ResNet50 backbone was **frozen**. Only the custom-added top layers (Head) were trained to map the pre-trained features to the specific classes (Mask vs. No Mask).

### Stage 2: Fine-Tuning
To specialize the model for facial features and mask textures, the **last 10 layers** of the ResNet50 architecture were **unfrozen**. The model was then re-trained with a lower learning rate. This stage significantly improved the model's ability to capture nuanced patterns, leading to a stabilized loss curve and higher accuracy.



---

## 📊 Results & Performance
* **Validation Accuracy:** 85%
* **Optimizer:** Adam
* **Learning Stability:** Observed a smooth convergence in the loss function post-fine-tuning, indicating effective weight adaptation without catastrophic forgetting.

---

## 🔗 Resources & Links
* **Live Application:** [View on Streamlit](https://ai-face-mask-detection1.streamlit.app/)
* **Demonstration Video:** [Watch on YouTube](https://youtu.be/E71ms9MDlOA)
* **Dataset:** [Download via Google Drive](https://drive.google.com/file/d/1a7Brk_hSxQPCAbPUF-qx0yPr-tlaljjK/view?usp=sharing)

---

## 📂 Project Structure
* `app.py`: Main Streamlit application script.
* `requirements.txt`: Environment dependencies.
* `models/`: Contains the trained model in `.keras` format.

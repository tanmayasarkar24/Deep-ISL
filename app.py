import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import pickle

# ================== File Paths ===================
MAIN_LOGO = r"C:\Users\TANMAYA\OneDrive\Desktop\DEEP_ISL\SIGN LANGUAGE LOGO.png"
ISL_STATIC_POSTER = r"C:\Users\TANMAYA\OneDrive\Desktop\DEEP_ISL\Static image.png"
ISL_STATIC_MODEL_PATH = r"C:\Users\TANMAYA\OneDrive\Desktop\DEEP_ISL\ISL Static\isl_cnn_model.h5"
ISL_STATIC_ENCODER_PATH = r"C:\Users\TANMAYA\OneDrive\Desktop\DEEP_ISL\ISL Static\label_encoder.pkl"

# ================== Load Model & Encoder ================
isl_static_model = load_model(ISL_STATIC_MODEL_PATH)
with open(ISL_STATIC_ENCODER_PATH, "rb") as f:
    isl_static_label_encoder = pickle.load(f)

# =================== ISL Static Prediction ===================
def predict_isl_static(image):
    img = image.convert('RGB').resize((48,48))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1,48,48,3)
    prediction = isl_static_model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    predicted_label = isl_static_label_encoder.inverse_transform([predicted_class])[0]
    return predicted_label, confidence

# ====================== Streamlit Modules ========================

def home_module():
    st.title("🖐️ Deep-ISL:Indian Sign Language Recognition (ISLR)")
    st.image(Image.open(MAIN_LOGO), use_column_width=True)
    st.markdown("""
        Welcome to Deep-ISL!  
        This project demonstrates Indian Sign Language (ISL) gesture recognition:
        - **ISL:** Recognize ISL alphabet and numbers from uploaded hand sign images.
        Select a module from the sidebar to begin.
    """)

def isl_static_module():
    st.header("Indian Sign Language (ISL) - Static Image Upload")
    st.image(Image.open(ISL_STATIC_POSTER), width=300)
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg","jpeg","png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Predict ISL Static"):
            pred_label, confidence = predict_isl_static(image)
            st.success(f"Prediction: {pred_label} (Confidence: {confidence:.2f})")

# =================== Main Routing ===========================
def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.selectbox("Choose Category 👇", [
        "Home",
        "ISL Static"
    ])
    if choice == "Home":
        home_module()
    elif choice == "ISL Static":
        isl_static_module()

if __name__ == "__main__":
    main()

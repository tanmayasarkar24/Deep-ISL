# Deep-ISL

# Hand-Speak: Indian Sign Language (ISL) Static Recognition


---

## About The Project

Hand-Speak is an Indian Sign Language (ISL) Static Gesture Recognition app that enables users to upload images of hand signs and get predictions of the corresponding ISL alphabet or number. This project demonstrates machine learning based ISL recognition using a CNN model trained on the [Indian Sign Language Dataset (ISLRTC) from Kaggle](https://www.kaggle.com/datasets/atharvadumbre/indian-sign-language-islrtc-referred).

The app is built with [Streamlit](https://streamlit.io/) and TensorFlow/Keras for easy deployment and interactive usage.

---

## Features

- Static ISL recognition from uploaded hand sign images.
- Lightweight CNN model trained on resized RGB images (48x48).
- User-friendly Streamlit interface with image upload and prediction display.
- Clear project structure focused on ISL static recognition only for stable performance.

---

## Dataset

This project uses the Indian Sign Language dataset from Kaggle:  
[https://www.kaggle.com/datasets/atharvadumbre/indian-sign-language-islrtc-referred](https://www.kaggle.com/datasets/atharvadumbre/indian-sign-language-islrtc-referred)

- Contains images for ISL alphabets (A-Z) and numbers (0-9).
- Dataset preprocessing involves resizing images to 48x48 RGB before model training.

---

## Getting Started

### Prerequisites

- Python 3.8 or above
- Streamlit
- TensorFlow
- Pillow
- NumPy
- scikit-learn
- OpenCV (for image processing)

### Installation
1. Clone this repository.
git clone https://github.com/tanmayasarkar24/Deep-ISL.git
cd Deep-ISL

2. Install required packages:
pip install -r requirements.txt

(You can generate `requirements.txt` with your environment’s packages.)

3. Download the ISL dataset from Kaggle and place it in the specified dataset folder if you plan to retrain or extend the model.

---

## Usage

Run the Streamlit app with:
streamlit run app.py


- On the home page, you will find a project overview.
- Navigate to **ISL Static** via the sidebar.
- Upload any hand sign image (jpg, jpeg, png).
- Click **Predict ISL Static** to see the model’s prediction with confidence.

---

## Model and Code Structure

- `app.py`: Streamlit app for UI and prediction.
- `isl_cnn_model.h5`: Pretrained CNN model for ISL static image classification.
- `label_encoder.pkl`: Label encoder for mapping model output to actual ISL classes.
- Images for UI stored locally in the project folder.

---

## Future Work

- Extend support for dynamic ISL gesture recognition with video input.
- Improve model accuracy with landmark-based training.
- Add real-time webcam support once model robustness improves.
- Incorporate more ISL signs and increase dataset size.

---

## Acknowledgements

- Dataset: Atharva Dumbre, Kaggle Indian Sign Language Dataset.
- Libraries: TensorFlow, Streamlit, MediaPipe.
- Inspired by open-source sign language recognition projects.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.




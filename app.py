import streamlit as st
import cv2
import numpy as np
from PIL import Image
import joblib
import tempfile

# Load models
clf = joblib.load("models/sift_pca_svm.pkl")
pca = joblib.load("models/pca_model.pkl")
le = joblib.load("models/label_encoder.pkl")
sift = cv2.SIFT_create()

def extract_sift_features_from_pil(pil_img):
    img = np.array(pil_img.convert("L"))  # grayscale
    keypoints, descriptors = sift.detectAndCompute(img, None)
    if descriptors is None:
        return None
    return descriptors.flatten()[:500]

st.title("🔎 CAPTCHA Recognition with SIFT + PCA + SVM")
uploaded_file = st.file_uploader("Upload CAPTCHA image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    features = extract_sift_features_from_pil(image)
    if features is not None:
        features_pca = pca.transform([features])
        pred = clf.predict(features_pca)
        pred_label = le.inverse_transform(pred)[0]
        st.success(f"Predicted Text: {pred_label}")
    else:
        st.error("No SIFT features found in this image.")

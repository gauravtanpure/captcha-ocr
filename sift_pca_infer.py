import cv2
import numpy as np
import joblib
import os

sift = cv2.SIFT_create()

def extract_sift_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = sift.detectAndCompute(img, None)
    if descriptors is None:
        return None
    return descriptors.flatten()[:500]

# Load models
clf = joblib.load("models/sift_pca_svm.pkl")
pca = joblib.load("models/pca_model.pkl")
le = joblib.load("models/label_encoder.pkl")

def predict_image(image_path):
    desc = extract_sift_features(image_path)
    if desc is None:
        return "No features found"
    desc_pca = pca.transform([desc])
    pred = clf.predict(desc_pca)
    return le.inverse_transform(pred)[0]

print(predict_image("dataset/test/2mg87.png"))

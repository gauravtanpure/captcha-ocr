import cv2
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Path
train_dir = "dataset/train"

# SIFT extractor
sift = cv2.SIFT_create()

def extract_sift_features(image_path):
    """Extract SIFT descriptors and flatten to fixed length."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    keypoints, descriptors = sift.detectAndCompute(img, None)
    if descriptors is None:
        return None
    return descriptors.flatten()[:500]  # limit to 500 values

def load_dataset(path):
    """Load all images and extract features."""
    X, y = [], []
    for fname in os.listdir(path):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            label = os.path.splitext(fname)[0]
            desc = extract_sift_features(os.path.join(path, fname))
            if desc is not None:
                X.append(desc)
                y.append(label)
    return np.array(X), np.array(y)

print("📥 Loading dataset...")
X, y = load_dataset(train_dir)
print(f"Total samples: {len(y)}")

# 🔑 Limit to N classes (subset)
MAX_CLASSES = 856
unique_labels = list(set(y))[:MAX_CLASSES]
mask = [label in unique_labels for label in y]
X, y = X[mask], y[mask]

print(f"Filtered samples: {len(y)} across {len(unique_labels)} classes")

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# PCA
n_components = min(100, X.shape[0], X.shape[1])
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

# Train classifier
clf = SVC(kernel="linear", probability=True)
clf.fit(X_pca, y_enc)

# Training accuracy
y_pred = clf.predict(X_pca)
acc = accuracy_score(y_enc, y_pred)
print(f"✅ Training Accuracy: {acc:.4f}")

# Save models
joblib.dump(clf, "models/sift_pca_svm.pkl")
joblib.dump(pca, "models/pca_model.pkl")
joblib.dump(le, "models/label_encoder.pkl")
print("💾 Models saved: sift_pca_svm.pkl, pca_model.pkl, label_encoder.pkl")

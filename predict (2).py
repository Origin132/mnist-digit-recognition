"""
MNIST Handwritten Digit Recognition
=====================================
Use this script to predict a handwritten digit
from the MNIST test set using the trained Neural Network.

Usage:
    python predict.py
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


# ── Load saved model ──────────────────────────────────────
model = load_model('mnist_model.keras')
print("✅ Model loaded successfully!")


# ── Load and preprocess MNIST test data ───────────────────
(_, _), (X_test, y_test) = mnist.load_data()

# Normalise pixel values to [0, 1]
X_test = X_test / 255.0

# One-hot encode labels
y_test_encoded = to_categorical(y_test, 10)


# ── Pick an image to predict ──────────────────────────────
# Change this index (0–9999) to test different images
IMAGE_INDEX = 5

image        = X_test[IMAGE_INDEX]
true_label   = y_test[IMAGE_INDEX]


# ── Make prediction ───────────────────────────────────────
prediction   = model.predict(image.reshape(1, 28, 28))
predicted_digit = np.argmax(prediction)
confidence      = np.max(prediction) * 100


# ── Display result ────────────────────────────────────────
print("\n" + "=" * 40)
print(f"  🔢 Predicted Digit : {predicted_digit}")
print(f"  ✅ Actual Digit    : {true_label}")
print(f"  📊 Confidence      : {confidence:.2f}%")

if predicted_digit == true_label:
    print("  🟢 Result          : CORRECT")
else:
    print("  🔴 Result          : WRONG")
print("=" * 40)


# ── Visualise the image ───────────────────────────────────
plt.figure(figsize=(3, 3))
plt.imshow(image, cmap="gray")
plt.title(f"Predicted: {predicted_digit}  |  Actual: {true_label}")
plt.axis("off")
plt.tight_layout()
plt.show()

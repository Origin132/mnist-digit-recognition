# 🔢 MNIST Handwritten Digit Recognition

A deep learning project that recognises handwritten digits (0–9) using a **Neural Network** built with TensorFlow and Keras, trained on the classic MNIST dataset.

---

## 📌 Project Overview

The MNIST dataset is the "Hello World" of deep learning — 70,000 images of handwritten digits used to benchmark image classification models. This project builds and trains a fully connected neural network that achieves **97.58% test accuracy** in under 10 training epochs.

| Item | Detail |
|---|---|
| **Model** | Sequential Neural Network (Dense layers) |
| **Framework** | TensorFlow / Keras |
| **Dataset** | MNIST (70,000 images — 60,000 train, 10,000 test) |
| **Input Shape** | 28 × 28 grayscale images |
| **Output Classes** | 10 (digits 0–9) |
| **Test Accuracy** | 97.58% |
| **Optimizer** | Adam |
| **Loss Function** | Categorical Crossentropy |
| **Epochs** | 10 |
| **Batch Size** | 32 |

---

## 📂 Project Structure

```
mnist-digit-recognition/
│
├── mnist_tensorflow.ipynb      # Full notebook (data loading + training + evaluation)
├── mnist_model.keras           # Saved trained model
├── predict.py                  # Script to make new predictions
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🧠 Model Architecture

```
Input: 28×28 grayscale image
        ↓
Flatten Layer       → 784 units
        ↓
Dense Layer         → 128 units, ReLU activation
        ↓
Dense Layer         → 64 units, ReLU activation
        ↓
Dense Output Layer  → 10 units, Softmax activation
        ↓
Output: Predicted digit (0–9)
```

---

## 📊 Training Results

| Epoch | Train Accuracy | Validation Accuracy |
|---|---|---|
| 1 | 91.93% | 95.52% |
| 2 | 96.24% | 96.85% |
| 3 | 97.53% | 97.12% |
| 5 | 98.47% | 97.19% |
| 8 | 99.29% | 97.70% |
| 10 | 99.51% | 97.53% |

**Final Test Accuracy: 97.58%**

---

## ⚙️ Data Preprocessing

- Loaded MNIST directly from `tensorflow.keras.datasets`
- Pixel values normalised from `[0, 255]` to `[0, 1]`
- Labels one-hot encoded using `to_categorical`
- 20% of training data used for validation split

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Origin132/mnist-digit-recognition.git
cd mnist-digit-recognition
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
Open `mnist_tensorflow.ipynb` in Jupyter or Google Colab and run all cells.

### 4. Make a new prediction
```bash
python predict.py
```

---

## 🛠️ Tech Stack

- Python 3.12
- TensorFlow / Keras
- NumPy
- Matplotlib

---

## 👤 Author

**Sambo Bashir**
Aspiring AI/ML Engineer | Deep Learning Practitioner

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/sambo-bashir-3a9648185/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/Origin132)

Update author links in README

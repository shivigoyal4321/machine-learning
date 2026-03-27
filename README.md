# 🧠 Machine Learning & Deep Learning Projects

This repository contains two end-to-end projects demonstrating practical implementations of classical machine learning and deep learning techniques using real-world datasets.

---

## 🔬 1. Tumor Classification Using Traditional ML

**Dataset**: [Breast Cancer Wisconsin Diagnostic Dataset (UCI)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))  
**Goal**: Predict whether a tumor is benign or malignant based on diagnostic features.

### 🔹 Highlights:
- Built a full ML pipeline from data preprocessing to model deployment.
- Implemented and compared 7 different algorithms: SVM, KNN, Random Forest, Logistic Regression, Decision Tree, Naive Bayes, and Gradient Boosting.
- Evaluated models on accuracy, efficiency, and complexity.
- Achieved **96.1% accuracy**, with detailed comparison across algorithms.
- Focused on model generalization, optimization, and interpretability.

**Tools & Libraries**:  
`Python`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `Google Colab`

📄 **Notebook**: `breast_cancer_model.ipynb`

---

## 🧮 2. Handwritten Digit Recognition with CNN

**Dataset**: [MNIST Handwritten Digit Dataset](http://yann.lecun.com/exdb/mnist/)  
**Goal**: Classify handwritten digits (0–9) using a Convolutional Neural Network.

### 🔹 Highlights:
- Designed a CNN architecture using TensorFlow and Keras for image classification.
- Achieved **98.7% validation accuracy**, with strong generalization on unseen data.
- Applied preprocessing (normalization), used ReLU activations and softmax for output.
- Visualized training performance via loss/accuracy plots and feature map activations.

**Tools & Libraries**:  
`Python`, `TensorFlow`, `Keras`, `Matplotlib`, `NumPy`

📄 **Notebook**: `mnist_cnn_model.ipynb`

---

## 📌 How to Use
1. Clone the repo:
   ```bash
   git clone https://github.com/shivigoyal4321/machine-learning.git
2. Launch notebooks in Jupyter or Google Colab.
Install required libraries:
pip install scikit-learn pandas matplotlib seaborn tensorflow keras
## Shivi Goyal – Machine Learning Enthusiast

## Deploy Breast Cancer API on Render

This repo now includes a production API for the breast cancer model.

### Files added
- `train_breast_cancer_model.py` - trains and exports `breast_cancer_model.pkl`
- `app.py` - Flask API (`/health`, `/schema`, `/predict`)
- `requirements.txt` - Python dependencies
- `Procfile` - Render process command
- `render.yaml` - optional Render Blueprint config

### Local run
```bash
python -m pip install -r requirements.txt
python train_breast_cancer_model.py
python app.py
```

### Render setup (Web Service)
1. Push this repository to GitHub.
2. In Render, click **New +** -> **Web Service** -> connect your repo.
3. Use:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
4. Deploy.

### API usage
- GET `/health`
- GET `/schema`
- POST `/predict`

Example request body:
```json
{
  "features": {
    "mean radius": 14.6,
    "mean texture": 22.7
  }
}
```

Note: send all required features listed by `/schema`.


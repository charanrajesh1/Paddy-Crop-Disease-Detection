# Paddy Crop Disease Detection Using Convolutional Neural Networks 

This repository contains the implementation of a deep learning–based image classification system for detecting diseases in paddy (rice) crops using Convolutional Neural Networks (CNNs).  
The project aims to support precision agriculture by enabling early, automated identification of rice leaf diseases from images.

---

##  Project Overview

Rice is a critical staple crop worldwide, but its yield is significantly affected by leaf diseases. Traditional disease diagnosis relies on manual inspection, which is time-consuming, subjective, and difficult to scale.

This project proposes a CNN-based framework to classify rice leaf images into six categories:

- Bacterial Leaf Blight  
- Brown Spot  
- Healthy Leaf  
- Leaf Blast  
- Leaf Scald  
- Sheath Blight  

The model is trained on a balanced dataset and achieves strong performance across multiple evaluation metrics, demonstrating its suitability for real-world agricultural applications.

---

## Methodology

### 1. Dataset
- Source: Rice Leaf Disease Dataset (Kaggle)
- Total Images: 3,829 RGB images
- Classes: 6 (balanced distribution)
- Images captured under varying lighting and background conditions

### 2. Preprocessing & Augmentation
- Image resizing to 128 × 128
- Pixel normalization to range [0, 1]
- Data augmentation techniques:
  - Rotation
  - Horizontal flipping
  - Zoom
  - Brightness adjustment

### 3. CNN Architecture
The network consists of:
- 3 Convolutional Blocks with ReLU activation and Max Pooling
- Dropout layers to prevent overfitting
- Fully connected dense layers
- Softmax output layer for multi-class classification

### 4. Training Configuration
- Optimizer: Adam
- Learning Rate: 0.001
- Loss Function: Categorical Cross-Entropy
- Batch Size: 32
- Epochs: Up to 30 (with Early Stopping)

---

## Results

| Metric | Value |
|------|------|
| Accuracy | ~95% |
| Macro F1-Score | ~0.96 |
| Weighted F1-Score | ~0.96 |

- Strong generalization across all disease classes  
- Minor confusion observed between visually similar diseases (e.g., Brown Spot vs Leaf Scald)  
- Healthy leaf class achieved the highest precision and recall  

---

## Tech Stack

- Programming Language: Python  
- Deep Learning: TensorFlow / Keras  
- Libraries: NumPy, OpenCV, Matplotlib, Scikit-learn  
- Environment: Jupyter Notebook / Python scripts  

---


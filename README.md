# Paddy Crop Disease Detection using CNN: a deep learning framework for sustainable agriculture

This project uses Deep Learning (TensorFlow/Keras) to detect and classify diseases in rice (paddy) crops from leaf images.

## 🎯 Aim of the Project
The primary goal of this project is to develop an automated system that can accurately identify various diseases affecting paddy crops. By using computer vision, this tool aims to assist farmers and agricultural experts in early disease diagnosis, potentially saving crops and increasing yield.

## 📌 Features
*   **Automatic Data Handling**: Downloads and extracts the dataset from Google Drive automatically.
*   **Space Efficient**: Supports downloading to a secondary drive (D:) to save space.
*   **Hardware Optimized**: Detects GPU capabilities and adjusts training epochs accordingly.
*   **Visualization**: Creates graphs for Training Accuracy/Loss and a sample prediction grid.

## ⚙️ Setup & Installation

1.  **Project Location**: Ensure you are in the project folder:
    `c:\Users\Modep\OneDrive\Desktop\paddy_crop_disesase_detection`

2.  **Dependencies**: Install the required Python packages:
    ```bash
    pip install tensorflow matplotlib gdown
    ```

## 🚀 Usage

Run the main automation script:

```bash
python colab_code.py
```

### What happens next?
1.  The script checks if `Rice_Leaf_AUG` dataset exists in `D:\paddy_disease_data`.
2.  If not, it downloads `archive.zip` and extracts it.
3.  It loads the images, builds a CNN model, and trains it.
4.  Finally, it saves the model and generates two result images:
    *   `training_results.png`
    *   `Sample_predictions.png`

## 🔗 Dataset Source
The dataset used in this project is available on Kaggle:
[Rice Disease Dataset - Anshulm257](https://www.kaggle.com/datasets/anshulm257/rice-disease-dataset)

## 📊 Dataset Classes
*   Bacterial Leaf Blight
*   Brown Spot
*   Healthy Rice Leaf
*   Leaf Blast
*   Leaf Scald
*   Sheath Blight

## 📁 Project Output
After running the script, the following outputs will be generated in the project directory:

1.  **Trained Model** (`paddy_disease_model.keras`):
    *   The complete trained CNN model saved in Keras format. You can load this later for making predictions on new images.

2.  **Training Visualization** (`training_results.png`):
    *   A graph plotting **Accuracy** and **Loss** over the training epochs.
    *   Use this to check if the model is learning correctly or overfitting.

3.  **Prediction Sample** (`sample_predictions.png`):
    *   A grid of 9 test images.
    *   Shows the **Predicted Label** vs the **True Label** for each image, giving you a visual sense of the model's accuracy.

## 📁 Files
*   `colab_code.py`: Complete Python script for the project.
*   `paddy_disease_model.keras`: Saved trained model.

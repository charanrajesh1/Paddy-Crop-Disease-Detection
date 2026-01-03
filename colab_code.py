import os
import sys
import zipfile
import warnings
import time
import numpy as np

# Auto-install gdown if missing
try:
    import gdown
except ImportError:
    print("Installing gdown...")
    os.system(f'{sys.executable} -m pip install gdown')
    import gdown

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter('ignore')

import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

# Configuration
BATCH_SIZE = 20
IMAGE_SIZE = 128
CHANNELS = 3

# Check GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(f"GPU detected: {physical_devices}")
    EPOCHS = 50
else:
    print("No GPU detected. Reducing EPOCHS to 3 for quick demonstration.")
    EPOCHS = 3

# ---------------------------------------------------------
# 1. Dataset Preparation
# ---------------------------------------------------------
zip_id = '16xyyeVqjw9kg8jwkaFHLg-tEbPTpXj0h'
# Use D: drive for storage
base_dir = r'D:\paddy_disease_data'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

zip_filename = os.path.join(base_dir, 'archive.zip')
extract_folder = os.path.join(base_dir, 'Rice_Leaf_AUG')

if not os.path.exists(extract_folder):
    if not os.path.exists(zip_filename):
        print(f"Downloading dataset (ID: {zip_id}) to {base_dir}...")
        try:
            url = f'https://drive.google.com/uc?id={zip_id}'
            gdown.download(url, zip_filename, quiet=False)
        except Exception as e:
            print(f"Download failed: {e}")
            sys.exit(1)

    print(f"Extracting dataset to {base_dir}...")
    try:
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(base_dir)
    except zipfile.BadZipFile:
        print("Error: Zip file is corrupt. Please delete it and try again.")
        sys.exit(1)
    print("Extraction complete.")
else:
    print(f"Dataset directory '{extract_folder}' already exists. Skipping download.")

# ---------------------------------------------------------
# 2. Data Loading
# ---------------------------------------------------------
print(f"Loading images from {extract_folder}...")
try:
    if not os.listdir(extract_folder):
         print(f"Directory {extract_folder} is empty. Re-extracting...")
         with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(base_dir)

    dataset = tf.keras.utils.image_dataset_from_directory(
        extract_folder,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
except Exception as e:
    print(f"Failed to load dataset: {e}")
    sys.exit(1)

class_names = dataset.class_names
print(f"Classes: {class_names}")

# Data Partitioning
def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    ds_size = len(ds)
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)
# Optimized pipeline (No cache to save space)
train_ds = train_ds.shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# ---------------------------------------------------------
# 3. Model Building
# ---------------------------------------------------------
print("Building CNN Model...")
input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = len(class_names)

model = models.Sequential([
    layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.Rescaling(1.0/255),
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax'),
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# ---------------------------------------------------------
# 4. Training
# ---------------------------------------------------------
print(f"Starting training for {EPOCHS} epochs...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    verbose=1,
    epochs=EPOCHS
)

# ---------------------------------------------------------
# 5. Results & Visualization
# ---------------------------------------------------------
print("Training complete. Saving results...")

# Save Model
model.save('paddy_disease_model.keras')
print("Model saved to paddy_disease_model.keras")

# 1. Training Graphs
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

save_path = 'training_results.png'
plt.savefig(save_path)
print(f"Training plots saved to {os.path.abspath(save_path)}")

# 2. Prediction Visualization
print("Generating sample predictions...")
plt.figure(figsize=(10, 10))
for images, labels in test_ds.take(1):
    predictions = model.predict(images)
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        
        pred_label = class_names[np.argmax(predictions[i])]
        true_label = class_names[labels[i]]
        
        plt.title(f"Pred: {pred_label}\nTrue: {true_label}")
        plt.axis("off")
        
pred_save_path = 'sample_predictions.png'
plt.savefig(pred_save_path)
print(f"Sample predictions saved to {os.path.abspath(pred_save_path)}")

final_acc = val_acc[-1]
print(f"Final Validation Accuracy: {final_acc:.4f}")
print("Script execution finished successfully.")

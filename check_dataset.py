import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import random

# Base path to your dataset
base_dir = r"C:\Pneumonia_Project\chest_xray"

# Paths for train, validation, and test
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

# Count number of images in each folder
for folder in ["train", "val", "test"]:
    normal = len(os.listdir(os.path.join(base_dir, folder, "NORMAL")))
    pneumonia = len(os.listdir(os.path.join(base_dir, folder, "PNEUMONIA")))
    print(f"{folder.upper()} → NORMAL: {normal},  PNEUMONIA: {pneumonia}")

# Load a few images to visually confirm
datagen = ImageDataGenerator(rescale=1./255)
sample_data = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=10,
    class_mode='binary'
)

# Show a few images from the dataset
images, labels = next(sample_data)
plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(images[i])
    plt.title("PNEUMONIA" if labels[i] == 1 else "NORMAL")
    plt.axis("off")
plt.tight_layout()
plt.show()

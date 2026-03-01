import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing import image
from tkinter import Tk, Label, Button, filedialog
from PIL import Image, ImageTk

# =======================================================
# --- Load model and class mapping ---
# =======================================================
MODEL_PATH = r"C:\Pneumonia_Project\pneumonia_mobilenetv2_final.h5"
CLASS_PATH = r"C:\Pneumonia_Project\class_indices.json"

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Load class indices (NORMAL=0, PNEUMONIA=1)
with open(CLASS_PATH, 'r') as f:
    class_indices = json.load(f)
idx2class = {v: k for k, v in class_indices.items()}

# =======================================================
# --- Prediction Function (fixed label logic) ---
# =======================================================
def predict_image(img_path):
    """
    Predicts the class (NORMAL or PNEUMONIA) for a given chest X-ray image.
    Model output is a single sigmoid value.
    For this model: high sigmoid value -> NORMAL
                    low sigmoid value  -> PNEUMONIA
    """
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    # Model prediction
    probs = model.predict(x)
    p = float(probs[0][0])

               # 🔁 Swapped again: higher sigmoid → PNEUMONIA
    label = "PNEUMONIA" if p > 0.5 else "NORMAL"
    confidence = p if label == "PNEUMONIA" else 1 - p

    return label, confidence


# =======================================================
# --- GUI Functionality ---
# =======================================================
def open_and_predict():
    filepath = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
    )
    if not filepath:
        return

    lbl_result.config(text="Predicting...", fg="blue")
    lbl_result.update()

    label, conf = predict_image(filepath)

    # Update result label
    color = "green" if label == "NORMAL" else "red"
    lbl_result.config(
        text=f"Result: {label} ({conf*100:.2f}%)",
        fg=color
    )

    # Show image preview
    img = Image.open(filepath).resize((250, 250))
    img_tk = ImageTk.PhotoImage(img)
    lbl_img.config(image=img_tk)
    lbl_img.image = img_tk


# =======================================================
# --- GUI Layout ---
# =======================================================
root = Tk()
root.title("Pneumonia Detection AI")
root.geometry("400x520")
root.configure(bg="#f0f0f0")

Label(root, text="Pneumonia Detection AI", font=("Arial", 16, "bold"), bg="#f0f0f0").pack(pady=15)

btn = Button(
    root,
    text="Select Chest X-ray",
    command=open_and_predict,
    font=("Arial", 12, "bold"),
    bg="#0078D7",
    fg="white",
    padx=10,
    pady=5
)
btn.pack(pady=10)

lbl_result = Label(root, text="No image selected", font=("Arial", 14), bg="#f0f0f0")
lbl_result.pack(pady=15)

lbl_img = Label(root, bg="#f0f0f0")
lbl_img.pack(pady=10)

Label(root, text="© Pneumonia Detection Project", font=("Arial", 9), bg="#f0f0f0", fg="gray").pack(side="bottom", pady=5)

root.mainloop()

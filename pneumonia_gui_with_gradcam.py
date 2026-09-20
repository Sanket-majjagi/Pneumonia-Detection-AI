# pneumonia_gui_with_gradcam.py
# Requires: tensorflow, pillow, numpy, opencv-python
# Run: python pneumonia_gui_with_gradcam.py

import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# ---------- CONFIG ----------
MODEL_PATH = "pneumonia_final_mobilenet.h5"
IMG_SIZE = (150, 150)   # same as training
HEATMAP_ALPHA = 0.5     # overlay transparency
# ----------------------------

# Load model
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    model = None
    load_error = str(e)

def preprocess_image(img_path, target_size=IMG_SIZE):
    img = image.load_img(img_path, target_size=target_size)
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_image(img_path):
    if model is None:
        raise RuntimeError("Model not loaded: " + load_error)
    x = preprocess_image(img_path)
    p = model.predict(x)[0][0]
    label = "PNEUMONIA" if p > 0.5 else "NORMAL"
    conf = float(p if p > 0.5 else 1.0 - p)
    return label, conf

# --------- Grad-CAM utilities ----------
def find_last_conv_layer(model):
    # Find last convolutional layer automatically (layer with 4D output and 'conv' in name preferred)
    for layer in reversed(model.layers):
        if hasattr(layer, "output_shape") and len(layer.output_shape) == 4:
            # prefer layers with 'conv' or 'block' naming, else first 4D encountered
            if "conv" in layer.name or "block" in layer.name or "project" in layer.name:
                return layer.name
    # fallback: last layer with 4D output
    for layer in reversed(model.layers):
        if hasattr(layer, "output_shape") and len(layer.output_shape) == 4:
            return layer.name
    raise ValueError("No 4D conv layer found in model.")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    img_array: preprocessed image batch (1, H, W, C)
    model: keras model
    last_conv_layer_name: string or None to auto-detect
    returns: heatmap (H, W) normalized [0..1]
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.outputs]
    )
    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = 0 if predictions.shape[-1] == 1 else tf.argmax(predictions[0])
        # For binary sigmoid output, predictions is shape (1,1) -> use predictions[0][0]
        loss = predictions[:, 0] if predictions.shape[-1] == 1 else predictions[:, pred_index]

    # Compute gradients of the top predicted class w.r.t. conv layer outputs
    grads = tape.gradient(loss, conv_outputs)
    # Pool gradients over spatial locations
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]  # (H, W, channels)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # weighted sum
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()
    return heatmap

def overlay_heatmap_on_image(orig_img_path, heatmap, output_path=None, alpha=HEATMAP_ALPHA, colormap=cv2.COLORMAP_JET):
    # Load original image in BGR (OpenCV) to overlay
    orig = cv2.imread(orig_img_path)
    if orig is None:
        raise FileNotFoundError("Cannot read image: " + orig_img_path)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    h, w = orig.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored_map = cv2.applyColorMap(heatmap_uint8, colormap)  # BGR
    colored_map = cv2.cvtColor(colored_map, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(colored_map, alpha, orig, 1 - alpha, 0)
    if output_path:
        # convert RGB to BGR for saving with cv2
        cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return overlay  # RGB numpy array

# ------------- GUI -------------
class PneumoniaApp:
    def __init__(self, master):
        self.master = master
        master.title("Pneumonia Detection — GUI with Grad-CAM")
        master.geometry("1000x600")
        master.resizable(False, False)

        # left frame: image canvas
        self.canvas_frame = tk.Frame(master, width=620, height=560, bd=2, relief="sunken")
        self.canvas_frame.place(x=10, y=10)
        self.canvas = tk.Label(self.canvas_frame)
        self.canvas.pack(expand=True)

        # right controls
        ctrl_x = 650
        tk.Label(master, text="Pneumonia Detection", font=("Helvetica", 16, "bold")).place(x=ctrl_x, y=10)

        self.open_btn = tk.Button(master, text="Open X-ray Image", width=20, command=self.open_image)
        self.open_btn.place(x=ctrl_x, y=60)

        self.predict_btn = tk.Button(master, text="Predict", width=20, state="disabled", command=self.run_predict)
        self.predict_btn.place(x=ctrl_x, y=100)

        self.gradcam_btn = tk.Button(master, text="Show Grad-CAM", width=20, state="disabled", command=self.show_gradcam)
        self.gradcam_btn.place(x=ctrl_x, y=140)

        self.save_btn = tk.Button(master, text="Save Result", width=20, state="disabled", command=self.save_result)
        self.save_btn.place(x=ctrl_x, y=180)

        self.path_text = tk.Text(master, width=40, height=2)
        self.path_text.place(x=ctrl_x, y=230)
        self.path_text.configure(state="disabled")

        tk.Label(master, text="Prediction:", font=("Arial", 10, "bold")).place(x=ctrl_x, y=300)
        self.pred_label = tk.Label(master, text="N/A", font=("Arial", 14))
        self.pred_label.place(x=ctrl_x, y=330)

        tk.Label(master, text="Confidence:", font=("Arial", 10, "bold")).place(x=ctrl_x, y=370)
        self.conf_label = tk.Label(master, text="N/A", font=("Arial", 12))
        self.conf_label.place(x=ctrl_x, y=395)

        tk.Label(master, text="Notes:", font=("Arial", 9, "italic")).place(x=ctrl_x, y=440)
        tk.Label(master, text="Grad-CAM highlights image areas used by the model.", justify="left").place(x=ctrl_x, y=460)

        self.current_image_path = None
        self.last_result = None
        self.last_gradcam_path = None

        if model is None:
            messagebox.showerror("Model Error", f"Failed to load model: {load_error}")
            self.open_btn.configure(state="disabled")

    def open_image(self):
        filetypes = [("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        path = filedialog.askopenfilename(initialdir=".", title="Select X-ray image", filetypes=filetypes)
        if not path:
            return
        self.current_image_path = path
        self.path_text.configure(state="normal")
        self.path_text.delete(1.0, tk.END)
        self.path_text.insert(tk.END, path)
        self.path_text.configure(state="disabled")

        img = Image.open(path).convert("RGB")
        img.thumbnail((600, 560))
        self.display_img = ImageTk.PhotoImage(img)
        self.canvas.configure(image=self.display_img)
        self.predict_btn.configure(state="normal")
        self.gradcam_btn.configure(state="disabled")
        self.save_btn.configure(state="disabled")
        self.pred_label.configure(text="N/A")
        self.conf_label.configure(text="N/A")
        self.last_result = None
        self.last_gradcam_path = None

    def run_predict(self):
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Choose an X-ray image first.")
            return
        try:
            label, conf = predict_image(self.current_image_path)
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            return
        self.pred_label.configure(text=label, fg="green" if label=="NORMAL" else "red")
        self.conf_label.configure(text=f"{conf*100:.2f}%")
        self.last_result = (self.current_image_path, label, conf)
        self.save_btn.configure(state="normal")
        self.gradcam_btn.configure(state="normal")  # allow gradcam after predicting

    def show_gradcam(self):
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Choose an X-ray image first.")
            return
        try:
            # preprocess image sized to IMG_SIZE
            arr = preprocess_image(self.current_image_path, target_size=IMG_SIZE)
            heatmap = make_gradcam_heatmap(arr, model)  # auto last conv layer
            # overlay on full-resolution original
            out_file = os.path.splitext(os.path.basename(self.current_image_path))[0] + "_gradcam.png"
            out_path = os.path.join(".", out_file)
            overlay = overlay_heatmap_on_image(self.current_image_path, heatmap, output_path=out_path, alpha=HEATMAP_ALPHA)
            self.last_gradcam_path = out_path

            # display overlay in GUI
            pil_img = Image.fromarray(overlay)
            pil_img.thumbnail((600, 560))
            self.display_img = ImageTk.PhotoImage(pil_img)
            self.canvas.configure(image=self.display_img)
            self.save_btn.configure(state="normal")
            messagebox.showinfo("Grad-CAM", f"Grad-CAM created and saved as:\n{out_path}")
        except Exception as e:
            messagebox.showerror("Grad-CAM Error", str(e))

    def save_result(self):
        if not (self.last_result or self.last_gradcam_path):
            messagebox.showwarning("No Result", "Predict or create Grad-CAM first.")
            return
        # create default text summary and allow saving
        base = os.path.basename(self.current_image_path)
        label = self.last_result[1] if self.last_result else "N/A"
        conf = self.last_result[2] if self.last_result else 0.0
        out_txt = f"Image: {base}\nPath: {self.current_image_path}\nPrediction: {label}\nConfidence: {conf*100:.2f}%\n"
        if self.last_gradcam_path:
            out_txt += f"Grad-CAM: {self.last_gradcam_path}\n"
        save_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")], initialfile=f"result_{base}.txt")
        if save_path:
            with open(save_path, "w") as f:
                f.write(out_txt)
            messagebox.showinfo("Saved", f"Result saved to:\n{save_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PneumoniaApp(root)
    root.mainloop()

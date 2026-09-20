# pneumonia_gui.py — Final integrated GUI with thresholded, smoothed Grad-CAM
# ==========================================================
# Pneumonia Detection AI — GUI vFinal
# ==========================================================
import os
import sys
import traceback
import json
import numpy as np
import tensorflow as tf
tf.config.run_functions_eagerly(True)  # avoid SymbolicTensor/.numpy() issues in analysis runs
from tensorflow.keras.preprocessing import image
from tkinter import Tk, Label, Button, filedialog, Canvas, Frame
from PIL import Image, ImageTk, ImageOps, ImageDraw, ImageFilter

# =======================================================
# --- Config / paths ---
# =======================================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pneumonia_mobilenetv2.h5")
CLASS_PATH = os.path.join(os.path.dirname(__file__), "class_indices.json")
CHOSEN_THRESH_PATH = os.path.join(os.path.dirname(__file__), "chosen_threshold.json")
INPUT_SIZE = (224, 224)
DISPLAY_SIZE = (340, 340)
HISTORY_LEN = 10

# =======================================================
# --- Helper utilities & imports/fallbacks ---
# =======================================================
# Pillow resampling compatibility
try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.ANTIALIAS

def safe_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

# =======================================================
# --- Load model safely with descriptive errors ---
# =======================================================
model = None
idx2class = None
try:
    safe_print(f"Trying to load model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    safe_print("✅ Model loaded successfully.")
except Exception:
    safe_print("❌ Error loading model:")
    traceback.print_exc()
    model = None

# Load class indices if present
try:
    if os.path.exists(CLASS_PATH):
        with open(CLASS_PATH, 'r') as f:
            class_indices = json.load(f)
        idx2class = {v: k for k, v in class_indices.items()}
        safe_print("✅ Class indices loaded.")
    else:
        safe_print("⚠️ class_indices.json not found, continuing without it.")
        idx2class = None
except Exception:
    safe_print("⚠️ Failed to load class indices (continuing).")
    idx2class = None

# =======================================================
# --- Threshold loading (chosen by analysis) ---
# =======================================================
THRESHOLD = 0.5
try:
    if os.path.exists(CHOSEN_THRESH_PATH):
        with open(CHOSEN_THRESH_PATH, "r") as f:
            tinfo = json.load(f)
            THRESHOLD = float(tinfo.get("best_threshold", 0.5))
            safe_print("Using THRESHOLD from chosen_threshold.json:", THRESHOLD)
    else:
        safe_print("No chosen_threshold.json found — using default THRESHOLD=0.5")
except Exception:
    safe_print("Error reading chosen_threshold.json — using default THRESHOLD=0.5")

# =======================================================
# --- Grad-CAM state & parameters ---
# =======================================================
last_image_path = None
last_prob = None
_gradcam_heatmap_cache = {}   # caches raw heatmaps (numpy 0..1)
_gradcam_on = False
_gradcam_img_tk = None

# Tunable conservative defaults
MIN_OVERLAY_PROB = 0.35      # show overlay only when p >= this
HEATMAP_THRESHOLD = 0.15    # suppress weak heatmap pixels
GAUSSIAN_BLUR_RADIUS = 4    # smooth heatmap
ROUND_RADIUS = 35

prob_history = [0.5] * HISTORY_LEN

# =======================================================
# --- Helper functions: image loading, display, predict ---
# =======================================================
def make_gradient(width, height, c1, c2):
    base = Image.new('RGB', (width, height), c1)
    top = Image.new('RGB', (width, height), c2)
    mask = Image.new('L', (width, height))
    for y in range(height):
        mask.putpixel((0, y), int(255 * (y / height)))
    mask = mask.resize((width, height))
    base.paste(top, (0, 0), mask)
    return ImageTk.PhotoImage(base)

def rounded_image(img, radius=30):
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, img.size[0], img.size[1]), radius, fill=255)
    img.putalpha(mask)
    return img

def safe_load_img(path, target_size):
    try:
        img = Image.open(path).convert("RGB")
        img = ImageOps.fit(img, target_size, RESAMPLE)
        return img
    except Exception:
        safe_print("❌ Error loading image:", path)
        traceback.print_exc()
        raise

def predict_image(img_path):
    """Return (label_str, pneumonia_prob 0..1)"""
    if model is None:
        raise RuntimeError("Model not loaded. Cannot predict.")
    x_img = image.load_img(img_path, target_size=INPUT_SIZE)
    x = image.img_to_array(x_img) / 255.0
    x = np.expand_dims(x, 0)
    preds = model.predict(x, verbose=0)
    preds = np.array(preds)
    safe_print("DEBUG: raw prediction output shape:", preds.shape, "values:", preds)
    # robust extraction
    try:
        if preds.ndim == 1:
            p = float(preds[0])
        elif preds.ndim == 2 and preds.shape[1] == 1:
            p = float(preds[0,0])
        elif preds.ndim == 2 and preds.shape[1] >= 2:
            p = float(preds[0,1])
        else:
            p = float(preds.reshape(-1)[0])
    except Exception:
        safe_print("❌ Unexpected prediction format:")
        traceback.print_exc()
        p = 0.0
    if np.isnan(p) or p is None:
        p = 0.0
    p = float(max(0.0, min(1.0, p)))
    label = "PNEUMONIA" if p >= THRESHOLD else "NORMAL"
    return label, p

# =======================================================
# --- Grad-CAM: compute, smooth, threshold, overlay ---
# =======================================================
def _find_last_conv_layer_name(m):
    for layer in reversed(m.layers):
        if "conv" in layer.__class__.__name__.lower():
            return layer.name
    return None

_last_conv = None
if model is not None:
    _last_conv = _find_last_conv_layer_name(model)
    if _last_conv is None:
        safe_print("Warning: No Conv layer found for Grad-CAM; Grad-CAM disabled.")
    else:
        safe_print("Grad-CAM will use conv layer:", _last_conv)

def _compute_raw_heatmap(img_array, eps=1e-8):
    """
    Grad-CAM++ implementation (drop-in replacement).
    img_array: numpy array shape (1,H,W,C) scaled 0..1
    returns: 2D numpy heatmap normalized to 0..1, or None on failure
    """
    try:
        if model is None or _last_conv is None:
            return None

        # small local model that outputs conv features + predictions
        grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(_last_conv).output, model.output])

        x = tf.convert_to_tensor(img_array, dtype=tf.float32)
        with tf.GradientTape() as tape:
            conv_outputs, preds = grad_model(x)
            # select score for 'pneumonia' class robustly
            if preds.shape.rank == 1 or (preds.shape.rank == 2 and preds.shape[1] == 1):
                score = preds[:, 0]
            else:
                score = preds[:, 1]

        # grads: shape (1, H, W, C)
        grads = tape.gradient(score, conv_outputs)
        if grads is None:
            return None

        # convert to float32 to avoid surprises
        conv_outputs = tf.cast(conv_outputs, tf.float32)  # (1,H,W,C)
        grads = tf.cast(grads, tf.float32)

        # Grad-CAM++ weights calculation
        grads_power_2 = tf.pow(grads, 2.0)                 # (1,H,W,C)
        grads_power_3 = tf.pow(grads, 3.0)                 # (1,H,W,C)

        # sum_over_spatial = sum_{i,j} conv_outputs_ij * grads_power_3_ij
        sum_spatial = tf.reduce_sum(conv_outputs * grads_power_3, axis=(1, 2), keepdims=True)  # (1,1,1,C)

        denominator = 2.0 * grads_power_2 + sum_spatial  # broadcasting (1,H,W,C) + (1,1,1,C)
        # avoid division by zero
        alpha = grads_power_2 / (denominator + eps)      # (1,H,W,C)

        # positive gradients (ReLU)
        positive_grads = tf.nn.relu(grads)               # (1,H,W,C)

        # weights: sum_{i,j} alpha_ij * ReLU(grad_ij)  -> shape (1, C)
        weights = tf.reduce_sum(alpha * positive_grads, axis=(1, 2))  # (1,C)

        # combine weights with conv outputs
        conv_outputs_squeezed = conv_outputs[0]          # (H,W,C)
        weights_squeezed = weights[0]                    # (C,)

        # compute cam = sum_k weights_k * conv_outputs[:,:,k]
        cam = tf.tensordot(conv_outputs_squeezed, weights_squeezed, axes=([2], [0]))  # (H, W)

        cam = tf.maximum(cam, 0.0)  # ReLU
        cam_np = cam.numpy()
        if cam_np.max() > eps:
            cam_np = cam_np / (cam_np.max() + eps)
        else:
            cam_np = np.zeros_like(cam_np, dtype=np.float32)

        return cam_np.astype(np.float32)

    except Exception:
        safe_print("Grad-CAM++ compute failed:")
        traceback.print_exc()
        return None


def _smooth_and_resize_heatmap(raw_heatmap, size=DISPLAY_SIZE):
    hm_img = Image.fromarray(np.uint8(raw_heatmap*255)).resize(size, resample=Image.BILINEAR)
    hm_img = hm_img.filter(ImageFilter.GaussianBlur(radius=GAUSSIAN_BLUR_RADIUS))
    hm = np.array(hm_img).astype("float32") / 255.0
    if hm.max() > 0:
        hm = hm / (hm.max() + 1e-8)
    return hm

# requires matplotlib for colormap
try:
    import matplotlib
    import matplotlib.cm as cm
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False
    safe_print("matplotlib not available — jet colormap will fallback to red-purple overlay.")

def _overlay_heatmap_on_pil_with_threshold(pil_img, heatmap, alpha=0.6, threshold=HEATMAP_THRESHOLD, percentile_keep=0.75):
    """
    Jet-style overlay:
    - keeps only the top (1 - percentile_keep) fraction of heatmap values
    - maps heatmap values -> jet colormap (if available)
    - blends with alpha mask scaled by heatmap value
    """
    try:
        if heatmap is None or heatmap.size == 0:
            return pil_img, False

        pct = float(np.clip(percentile_keep, 0.0, 0.99))
        dyn_thresh = max(float(threshold), float(np.quantile(heatmap, pct)))
        if dyn_thresh >= 1.0:
            return pil_img, False

        mask = (heatmap - dyn_thresh) / (1.0 - dyn_thresh)
        mask = np.clip(mask, 0.0, 1.0)
        if mask.max() < 1e-5:
            return pil_img, False

        alpha = float(alpha)
        alpha = min(alpha, 0.6)  # prevent full opacity

        # Resize mask to image size
        mask_img = Image.fromarray(np.uint8(mask * 255)).resize(pil_img.size, resample=RESAMPLE)
        mask_np = np.array(mask_img).astype("float32") / 255.0  # 0..1

        # Create colored map: use matplotlib's jet if available, otherwise fallback to red-purple
        if _HAS_MPL:
            cmap = cm.get_cmap("jet")
            # cmap expects values 0..1 -> returns RGBA floats 0..1
            colored = (cmap(mask)[:, :, :3] * 255).astype(np.uint8)
        else:
            # fallback: red->purple mapping
            r = (255.0 * mask).astype(np.uint8)
            g = (80.0 * (1.0 - mask)).astype(np.uint8)
            b = (255.0 * (1.0 - mask)).astype(np.uint8)
            colored = np.stack([r, g, b], axis=-1)

        # Convert to PIL and resize to match image
        overlay = Image.fromarray(colored).resize(pil_img.size, resample=RESAMPLE)

        # Build per-pixel alpha from mask * alpha
        alpha_mask = Image.fromarray(np.uint8(mask_np * 255.0 * alpha))

        overlay.putalpha(alpha_mask)
        base = pil_img.convert("RGBA")
        blended = Image.alpha_composite(base, overlay)
        return blended.convert("RGB"), True

    except Exception:
        safe_print("Error in jet overlay generation:")
        traceback.print_exc()
        return pil_img, False


def _get_cached_heatmap(path):
    if path in _gradcam_heatmap_cache:
        return _gradcam_heatmap_cache[path]
    try:
        pil = safe_load_img(path, INPUT_SIZE)
        arr = np.array(pil).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=0)
        raw = _compute_raw_heatmap(arr)
        if raw is None:
            return None
        smooth = _smooth_and_resize_heatmap(raw, DISPLAY_SIZE)
        _gradcam_heatmap_cache[path] = smooth
        return smooth
    except Exception:
        safe_print("Grad-CAM compute failed for:", path)
        traceback.print_exc()
        return None

def _generate_and_show_gradcam():
    global _gradcam_img_tk, last_image_path, last_prob
    if last_image_path is None:
        safe_print("Grad-CAM: no image loaded.")
        return
    p = 0.0 if last_prob is None else float(last_prob)
    if p < MIN_OVERLAY_PROB:
        safe_print(f"Grad-CAM suppressed (p={p:.3f} < {MIN_OVERLAY_PROB}).")
        _restore_original_display()
        return
    hm = _get_cached_heatmap(last_image_path)
    if hm is None:
        safe_print("No heatmap available for:", last_image_path)
        return
    scale = max(0.0, (p - MIN_OVERLAY_PROB) / (1.0 - MIN_OVERLAY_PROB))
    alpha = 0.25 + 0.75 * scale
    overlay, shown = _overlay_heatmap_on_pil_with_threshold(safe_load_img(last_image_path, DISPLAY_SIZE), hm, alpha=alpha, threshold=HEATMAP_THRESHOLD)
    if not shown:
        safe_print("Grad-CAM thresholding removed all highlights (no overlay).")
        _restore_original_display()
        return
    _gradcam_img_tk = ImageTk.PhotoImage(rounded_image(overlay.copy(), ROUND_RADIUS))
    canvas_img.itemconfig(img_container, image=_gradcam_img_tk)
    canvas_img.image = _gradcam_img_tk
    safe_print(f"Grad-CAM shown (alpha={alpha:.3f}, p={p:.3f}).")

def _restore_original_display():
    global last_image_path
    if last_image_path is None:
        return
    try:
        img = safe_load_img(last_image_path, DISPLAY_SIZE)
        img = rounded_image(img, ROUND_RADIUS)
        img_tk = ImageTk.PhotoImage(img)
        canvas_img.itemconfig(img_container, image=img_tk)
        canvas_img.image = img_tk
    except Exception:
        safe_print("Failed to restore original image display.")
        traceback.print_exc()

def toggle_gradcam():
    global _gradcam_on
    if model is None:
        safe_print("Model not loaded — cannot compute Grad-CAM.")
        return
    if last_image_path is None:
        safe_print("No image loaded to generate Grad-CAM.")
        return
    _gradcam_on = not _gradcam_on
    safe_print("Grad-CAM toggled:", _gradcam_on)
    if _gradcam_on:
        _generate_and_show_gradcam()
    else:
        _restore_original_display()

# =======================================================
# --- Visualization helpers (bars, sparkline, confidence) ---
# =======================================================
def draw_bars(p_norm, p_pneu):
    try:
        bar_canvas.delete("all")
        w, h = 100, 220
        pad = 40
        nh, ph = int(h * p_norm), int(h * p_pneu)
        bar_canvas.create_rectangle(pad, h-nh+pad, pad+w, h+pad, fill="#2ECC71", outline="#1E8449", width=1)
        bar_canvas.create_text(pad+w/2, h+pad+20, text=f"NORMAL\n{p_norm*100:.1f}%", font=("Helvetica", 10), fill="#134E4A")
        bar_canvas.create_rectangle(pad*2+w, h-ph+pad, pad*2+2*w, h+pad, fill="#E74C3C", outline="#A93226", width=1)
        bar_canvas.create_text(pad*2+1.5*w, h+pad+20, text=f"PNEUMONIA\n{p_pneu*100:.1f}%", font=("Helvetica", 10), fill="#5A1313")
    except Exception:
        safe_print("Warning: draw_bars failed.")
        traceback.print_exc()

def draw_sparkline():
    try:
        spark_canvas.delete("all")
        w, h, m = 260, 40, 5
        data = prob_history[-HISTORY_LEN:]
        xs = np.linspace(m, w-m, len(data))
        ys = [m + (1 - v)*(h-2*m) for v in data]
        spark_canvas.create_rectangle(0, 0, w, h, fill="#FFFFFF", outline="#DCEFF6")
        pts = []
        for x, y in zip(xs, ys):
            pts.extend([x, y])
        spark_canvas.create_line(pts, fill="#3498DB", width=2, smooth=True)
        for x, y, v in zip(xs, ys, data):
            color = "#E74C3C" if v > 0.6 else "#2ECC71"
            spark_canvas.create_oval(x-3, y-3, x+3, y+3, fill=color, outline="")
    except Exception:
        safe_print("Warning: draw_sparkline failed.")
        traceback.print_exc()

def draw_confidence_bar(p, mode="default"):
    try:
        conf_canvas.delete("all")
        w, h = 360, 28
        conf_canvas.create_rectangle(2, 2, w-2, h-2, fill="#FFFFFF", outline="#D6EAF8", width=1)
        fill_w = int((w-4) * p)
        if mode == "default":
            start_color = (52, 152, 219)
            end_color = (155, 89, 182)
        elif mode == "normal":
            start_color = (46, 204, 113)
            end_color = (52, 152, 219)
        else:
            start_color = (231, 76, 60)
            end_color = (52, 152, 219)
        if fill_w > 0:
            for i in range(fill_w):
                r = int(start_color[0] + (end_color[0] - start_color[0]) * (i / fill_w))
                g = int(start_color[1] + (end_color[1] - start_color[1]) * (i / fill_w))
                b = int(start_color[2] + (end_color[2] - start_color[2]) * (i / fill_w))
                conf_canvas.create_line(3+i, 3, 3+i, h-3, fill=f"#{r:02x}{g:02x}{b:02x}")
        conf_canvas.create_text(w//2, h//2, text=f"{p*100:.1f}%", fill="#0B2E4A", font=("Helvetica", 11, "bold"))
    except Exception:
        safe_print("Warning: draw_confidence_bar failed.")
        traceback.print_exc()

# =======================================================
# --- Result display helpers ---
# =======================================================
def clear_result_frame():
    for w in result_frame.winfo_children():
        w.destroy()

def show_result_normal():
    clear_result_frame()
    Label(result_frame, text="Result:", fg="#1DA2D8", bg="#FFFFFF", font=("Helvetica", 18, "bold")).pack()
    Label(result_frame, text="NORMAL", fg="#2ECC71", bg="#FFFFFF", font=("Helvetica", 22, "bold")).pack(pady=5)

def show_result_pneumonia(p):
    clear_result_frame()
    line1 = Frame(result_frame, bg="#FFFFFF")
    line1.pack(anchor="w")
    Label(line1, text="Result:", fg="#1DA2D8", bg="#FFFFFF", font=("Helvetica", 18, "bold")).pack(side="left")
    Label(line1, text="  PNEUMONIA DETECTED", fg="#E53935", bg="#FFFFFF", font=("Helvetica", 20, "bold")).pack(side="left")
    Label(line1, text=" ⚠️", fg="black", bg="yellow", font=("Helvetica", 18, "bold")).pack(side="left", padx=(6,0))
    line2 = Frame(result_frame, bg="#FFFFFF")
    line2.pack(anchor="w", pady=(6,0))
    Label(line2, text="Pneumonia likelihood ", fg="#E67E22", bg="#FFFFFF", font=("Helvetica", 16, "bold")).pack(side="left")
    Label(line2, text="(AI confidence): ", fg="#154360", bg="#FFFFFF", font=("Helvetica", 15, "bold")).pack(side="left")
    Label(line2, text=f"{p*100:.1f}%", fg="#2ECC71", bg="#FFFFFF", font=("Helvetica", 17, "bold")).pack(side="left")

# =======================================================
# --- Prediction trigger with debug messages ---
# =======================================================
def open_and_predict():
    try:
        filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if not filepath:
            safe_print("No file selected.")
            return

        global last_image_path, _gradcam_on, last_prob
        last_image_path = filepath
        _gradcam_on = False

        clear_result_frame()
        Label(result_frame, text="Analyzing...", bg="#FFFFFF", fg="#555", font=("Helvetica", 18)).pack()
        root.update()

        safe_print("Loading image for prediction:", filepath)
        try:
            _ = safe_load_img(filepath, INPUT_SIZE)
        except Exception:
            safe_print("Image load warning — continuing to try model.predict (TensorFlow loader will attempt).")

        label, p_pneu = predict_image(filepath)
        safe_print(f"Prediction done. label={label}, p_pneu={p_pneu:.6f}")

        # store last probability for Grad-CAM decisions
        last_prob = float(p_pneu)

        p_norm = max(0.0, 1.0 - p_pneu)
        prob_history.pop(0)
        prob_history.append(p_pneu)

        # show image in left pane
        img = safe_load_img(filepath, DISPLAY_SIZE)
        img = rounded_image(img, ROUND_RADIUS)
        img_tk = ImageTk.PhotoImage(img)
        canvas_img.itemconfig(img_container, image=img_tk)
        canvas_img.image = img_tk

        # display result using chosen threshold and 'needs review' band
        if p_pneu >= THRESHOLD:
            show_result_pneumonia(p_pneu)
            mode = "pneumonia"
            conf_val = p_pneu
        elif p_pneu >= max(0.45, THRESHOLD * 0.75):
            clear_result_frame()
            Label(result_frame, text="Result:", fg="#1DA2D8", bg="#FFFFFF", font=("Helvetica", 18, "bold")).pack()
            Label(result_frame, text="Needs review — low confidence", fg="#E67E22", bg="#FFFFFF", font=("Helvetica", 16, "bold")).pack(pady=6)
            Label(result_frame, text=f"Pneumonia probability: {p_pneu*100:.1f}%", fg="#154360", bg="#FFFFFF", font=("Helvetica", 14)).pack()
            mode = "ambiguous"
            conf_val = p_pneu
        else:
            show_result_normal()
            mode = "normal"
            conf_val = p_norm

        draw_bars(p_norm, p_pneu)
        draw_sparkline()
        draw_confidence_bar(conf_val, mode=mode)

    except Exception:
        safe_print("❌ Error during open_and_predict:")
        traceback.print_exc()
        clear_result_frame()
        Label(result_frame, text="Error during prediction. See console for details.", fg="red", bg="#FFFFFF").pack()

# =======================================================
# --- GUI Layout ---
# =======================================================
root = Tk()
root.title("🩺 Pneumonia Detection AI")
root.geometry("1150x740")
root.minsize(900, 640)
root.resizable(True, True)

bg_img = make_gradient(1150, 740, "#DDF6F6", "#FFFFFF")
Label(root, image=bg_img).place(x=0, y=0, relwidth=1, relheight=1)

Label(root, text="Pneumonia Detection AI", font=("Helvetica", 30, "bold"), bg="#DDF6F6", fg="#1B2631").pack(pady=(18, 6))

main_frame = Frame(root, bg="#EAF6F6")
main_frame.pack(fill="both", expand=True, padx=30, pady=(0, 6))
main_frame.columnconfigure(0, weight=3)
main_frame.columnconfigure(1, weight=2)

# LEFT
left_frame = Frame(main_frame, bg="#FFFFFF", bd=2, relief="groove", highlightthickness=3, highlightbackground="#AED6F1")
left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
canvas_img = Canvas(left_frame, bg="white", highlightthickness=0)
canvas_img.pack(expand=True, fill="both", padx=20, pady=20)
img_container = canvas_img.create_image(300, 220, anchor="center")

# RIGHT
right_frame = Frame(main_frame, bg="#FFFFFF", bd=2, relief="groove", highlightthickness=3, highlightbackground="#AED6F1")
right_frame.grid(row=0, column=1, sticky="nsew", padx=(12, 0))

result_frame = Frame(right_frame, bg="#FFFFFF")
result_frame.pack(pady=14)

conf_canvas = Canvas(right_frame, width=360, height=28, bg="#FFFFFF", highlightthickness=0)
conf_canvas.pack(pady=(6, 10))

bar_canvas = Canvas(right_frame, width=340, height=240, bg="#FFFFFF", highlightthickness=0)
bar_canvas.pack(pady=(6, 10))

Label(right_frame, text="Recent Pneumonia Probability", font=("Helvetica", 11), bg="#FFFFFF", fg="#555").pack(pady=(8, 2))
spark_canvas = Canvas(right_frame, width=300, height=40, bg="#FFFFFF", highlightthickness=0)
spark_canvas.pack()

# BOTTOM: button
bottom_frame = Frame(root, bg="#DDF6F6")
bottom_frame.pack(fill="x", pady=(4,6))
btn = Button(bottom_frame, text="📂  Select Chest X-ray Image", command=open_and_predict,
             font=("Helvetica", 16, "bold"), bg="#3498DB", fg="white",
             activebackground="#2980B9", activeforeground="white", relief="flat", padx=25, pady=10)
btn.pack(pady=6)
gradcam_btn = Button(bottom_frame, text="🛰️ Toggle Grad-CAM", command=toggle_gradcam,
                     font=("Helvetica", 12, "bold"), bg="#F39C12", fg="white",
                     activebackground="#D68910", relief="flat", padx=12, pady=8)
gradcam_btn.pack(pady=(0,6))

footer = Label(root, text="© 2025 AI Pneumonia Detection Project — Grad-CAM shows model attention, not clinical diagnosis", font=("Helvetica", 9), bg="#DDF6F6", fg="#7F8C8D")
footer.pack(side="bottom", pady=(0, 6))

# Initialize visuals
try:
    draw_bars(0.5, 0.5)
    draw_sparkline()
    draw_confidence_bar(0.5, mode="default")
except Exception:
    safe_print("Warning: initialization drawing failed.")
    traceback.print_exc()

root.mainloop()

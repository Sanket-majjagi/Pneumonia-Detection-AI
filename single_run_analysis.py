# single_run_analysis.py
# Single-run: find threshold, generate smoothed Grad-CAM overlays, save summary
import os, json, csv, sys, traceback
import numpy as np
from glob import glob
from PIL import Image, ImageFilter, ImageOps
import tensorflow as tf
tf.config.run_functions_eagerly(True)   # <<< add this
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve
import math

# -------- CONFIG (edit paths if needed) --------
ROOT = r"C:\Pneumonia_Project"
MODEL_PATH = os.path.join(ROOT, "pneumonia_mobilenetv2_final.h5")
TEST_DIR = os.path.join(ROOT, "chest_xray", "test")
TARGET_SIZE = (224,224)       # model input size
DISPLAY_SIZE = (340,340)      # overlay image size saved
BATCH_SIZE = 32
OUT_DIR = os.path.join(ROOT, "gradcam_outputs")
CHOSEN_THRESHOLD_JSON = os.path.join(ROOT, "chosen_threshold.json")
SUMMARY_CSV = os.path.join(OUT_DIR, "gradcam_summary.csv")
MAX_IMAGES = None  # set to integer to limit processed images (None = all)
MIN_OVERLAY_PROB = 0.25  # don't show overlay for p < this
HEATMAP_THRESHOLD = 0.12 # keep heatmap pixels > this after blur
GAUSSIAN_BLUR_RADIUS = 3

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading model:", MODEL_PATH)
model = load_model(MODEL_PATH)
print("Loaded model. TensorFlow version:", tf.__version__)

# ---- helper: predict probs on test generator to compute best threshold ----
print("Preparing test generator for threshold calculation...")
test_datagen = ImageDataGenerator(rescale=1./255)
tg = test_datagen.flow_from_directory(
    TEST_DIR, target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=False
)
y_true = tg.classes
print("Test samples:", tg.samples, "Class indices:", tg.class_indices)

steps = int(np.ceil(tg.samples / tg.batch_size))
print("Running model.predict on generator (this may take a while)...")
preds = model.predict(tg, steps=steps, verbose=1)

# derive pneumonia score (robust)
if preds.ndim==2 and preds.shape[1] >= 2:
    y_scores = preds[:,1]
else:
    y_scores = preds.ravel()

# ROC, Youden's J
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
youden = tpr - fpr
best_idx = int(np.argmax(youden))
best_thresh = float(thresholds[best_idx])
sensitivity = float(tpr[best_idx])
specificity = float(1 - fpr[best_idx])

print(f"Best threshold by Youden's J = {best_thresh:.4f}")
print(f"Sensitivity (recall) = {sensitivity:.4f}, Specificity = {specificity:.4f}")

# save chosen threshold
with open(CHOSEN_THRESHOLD_JSON, "w") as f:
    json.dump({"best_threshold": best_thresh, "sensitivity": sensitivity, "specificity": specificity}, f, indent=2)
print("Saved chosen_threshold.json ->", CHOSEN_THRESHOLD_JSON)

# ---- Grad-CAM helpers ----
def find_last_conv_layer(m):
    for layer in reversed(m.layers):
        if "conv" in layer.__class__.__name__.lower():
            return layer.name
    return None

last_conv = find_last_conv_layer(model)
if last_conv is None:
    print("WARNING: No Conv layer found for Grad-CAM. Exiting.")
    sys.exit(1)
print("Using last conv layer:", last_conv)

# gradcam compute
@tf.function
# ---- replace previous @tf.function _gradcam_tf with this eager version ----
def _gradcam_tf(img_tensor):
    """
    Eager Grad-CAM computation. img_tensor expected shape (1,H,W,C) as numpy or tf.Tensor.
    Returns 2D numpy heatmap normalized to [0..1].
    """
    # Build small model that outputs conv features + predictions
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv).output, model.output])

    img_t = tf.convert_to_tensor(img_tensor, dtype=tf.float32)
    with tf.GradientTape() as tape:
        # ensure tape watches the conv layer outputs by watching the input
        tape.watch(img_t)
        conv_outputs, predictions = grad_model(img_t)
        # handle prediction shapes robustly
        if predictions.shape.rank == 1 or (predictions.shape.rank == 2 and predictions.shape[1] == 1):
            score = predictions[:, 0]
        else:
            # pneumonia class assumed index 1
            score = predictions[:, 1]

    grads = tape.gradient(score, conv_outputs)
    # If grads is None (rare), return zero heatmap
    if grads is None:
        return np.zeros((conv_outputs.shape[1], conv_outputs.shape[2]), dtype=np.float32)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # shape: (channels,)
    conv_outputs = conv_outputs[0]  # (H, W, channels)
    # Weight channels by pooled gradients
    weighted = conv_outputs * pooled_grads[tf.newaxis, tf.newaxis, :]
    heatmap = tf.reduce_sum(weighted, axis=-1).numpy()  # (H, W) numpy array

    # ReLU and normalize
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 1e-8:
        heatmap = heatmap / (heatmap.max() + 1e-8)
    else:
        heatmap = np.zeros_like(heatmap, dtype=np.float32)
    return heatmap


def smooth_and_resize_heatmap(heatmap, size=DISPLAY_SIZE, blur_radius=GAUSSIAN_BLUR_RADIUS):
    hm_img = Image.fromarray(np.uint8(heatmap*255)).resize(size, resample=Image.BILINEAR)
    hm_img = hm_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    hm = np.array(hm_img).astype("float32") / 255.0
    if hm.max() > 0:
        hm = hm / (hm.max() + 1e-8)
    return hm

def overlay_on_image(pil_img, heatmap, alpha=0.5, threshold=HEATMAP_THRESHOLD):
    mask = (heatmap > threshold).astype("float32") * heatmap
    if mask.max() < 1e-6:
        return pil_img, False
    hm_img = Image.fromarray(np.uint8(mask*255)).convert("L")
    red = Image.new("RGBA", pil_img.size, (255,0,0,0))
    def alpha_map(px): return int(px * 255 * alpha)
    red_mask = hm_img.point(alpha_map)
    red.putalpha(red_mask)
    base = pil_img.convert("RGBA")
    blended = Image.alpha_composite(base, red)
    return blended.convert("RGB"), True

# ---- iterate over test images (use generator file list for consistent ordering) ----
print("Building list of test image file paths...")
file_paths = []
labels = []
for cls_name, cls_index in tg.class_indices.items():
    folder = os.path.join(TEST_DIR, cls_name)
    for ext in ("*.jpeg","*.jpg","*.png","*.bmp"):
        file_paths += glob(os.path.join(folder, ext))
    labels += [cls_index] * len(glob(os.path.join(folder, "*.jpg")))  # not used; we'll derive labels by matching later

# safer approach: use tg.filenames which are relative paths in the same order as tg.classes
file_paths = [os.path.join(TEST_DIR, p) for p in tg.filenames]
y_true_ordered = list(tg.classes)

if MAX_IMAGES:
    file_paths = file_paths[:MAX_IMAGES]
    y_true_ordered = y_true_ordered[:MAX_IMAGES]

# create CSV
csv_rows = []
print("Processing images and generating overlays... (this may take time)")
for i, (fp, true_label) in enumerate(zip(file_paths, y_true_ordered)):
    try:
        # load, preprocess, predict single image
        pil = Image.open(fp).convert("RGB")
        pil_proc = pil.resize(TARGET_SIZE, resample=Image.BILINEAR)
        arr = np.array(pil_proc).astype("float32") / 255.0
        x = np.expand_dims(arr, axis=0)
        pred = model.predict(x, verbose=0)
        if pred.ndim==2 and pred.shape[1]>=2:
            p_pneu = float(pred[0,1])
        else:
            p_pneu = float(pred.reshape(-1)[0])
        pred_label = 1 if p_pneu >= best_thresh else 0

        # compute heatmap and overlay if needed
        heatmap = _gradcam_tf(x)  # 2D normalized
        hm_s = smooth_and_resize_heatmap(heatmap, DISPLAY_SIZE)
        # choose alpha scaled by prob (only if above MIN_OVERLAY_PROB)
        overlay_shown = False
        overlay_img = None
        if p_pneu >= MIN_OVERLAY_PROB:
            alpha = min(0.8, 0.25 + 0.75 * ((p_pneu - MIN_OVERLAY_PROB) / (1.0 - MIN_OVERLAY_PROB)))
            disp_pil = pil.resize(DISPLAY_SIZE, resample=Image.BILINEAR)
            overlay_img, overlay_shown = overlay_on_image(disp_pil, hm_s, alpha=alpha, threshold=HEATMAP_THRESHOLD)
            if overlay_shown:
                out_name = os.path.join(OUT_DIR, f"overlay_{i:04d}_{os.path.basename(fp)}")
                overlay_img.save(out_name)
        # always also save a copy of the original resized image for reference
        orig_out = os.path.join(OUT_DIR, f"orig_{i:04d}_{os.path.basename(fp)}")
        pil.resize(DISPLAY_SIZE, resample=Image.BILINEAR).save(orig_out)

        # compute some heatmap stats
        hm_mean = float(np.mean(hm_s))
        hm_max = float(np.max(hm_s))

        csv_rows.append({
            "index": i,
            "file": fp,
            "true_label": int(true_label),
            "pred_label": int(pred_label),
            "prob": float(p_pneu),
            "overlay_shown": int(bool(overlay_shown)),
            "heatmap_mean": hm_mean,
            "heatmap_max": hm_max,
            "orig_saved": orig_out,
            "overlay_saved": out_name if overlay_shown else ""
        })

        if (i+1) % 50 == 0:
            print(f"Processed {i+1}/{len(file_paths)} images...")
    except Exception:
        print("Error processing:", fp)
        traceback.print_exc()

# write CSV
keys = ["index","file","true_label","pred_label","prob","overlay_shown","heatmap_mean","heatmap_max","orig_saved","overlay_saved"]
with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    for r in csv_rows:
        writer.writerow(r)
print("Saved summary CSV:", SUMMARY_CSV)
print("Sample overlays (if created) are in:", OUT_DIR)
print("Done.")

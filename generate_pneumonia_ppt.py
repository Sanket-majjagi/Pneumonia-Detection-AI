from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ==============================================================
# Pneumonia Detection Using Deep Learning (MobileNetV2)
# Phase-1 Mini Project Presentation Generator
# ==============================================================

prs = Presentation()
prs.slide_height = Inches(7.5)
prs.slide_width = Inches(13.33)

def add_slide(title, points):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = RGBColor(245, 252, 255)

    top_rect = slide.shapes.add_shape(
        autoshape_type_id=1, left=Inches(0), top=Inches(0),
        width=Inches(13.33), height=Inches(1)
    )
    top_rect.fill.solid()
    top_rect.fill.fore_color.rgb = RGBColor(0, 153, 153)

    txBox = slide.shapes.add_textbox(Inches(0.5), Inches(0.2), Inches(12), Inches(1))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(30)
    p.font.bold = True
    p.font.color.rgb = RGBColor(255, 255, 255)

    content_box = slide.shapes.add_textbox(Inches(1), Inches(1.5), Inches(11.5), Inches(5.5))
    tf2 = content_box.text_frame
    for i, point in enumerate(points):
        p = tf2.add_paragraph() if i > 0 else tf2.paragraphs[0]
        p.text = point
        p.font.size = Pt(20)
        p.font.color.rgb = RGBColor(30, 60, 90)

# ---------------------- TITLE SLIDE --------------------------
slide = prs.slides.add_slide(prs.slide_layouts[6])
fill = slide.background.fill
fill.solid()
fill.fore_color.rgb = RGBColor(230, 255, 255)

tBox = slide.shapes.add_textbox(Inches(1), Inches(2.5), Inches(12), Inches(2))
tFrame = tBox.text_frame
p = tFrame.paragraphs[0]
p.text = "Pneumonia Detection Using Deep Learning (MobileNetV2)"
p.font.size = Pt(44)
p.font.bold = True
p.font.color.rgb = RGBColor(0, 102, 102)
p.alignment = PP_ALIGN.CENTER

sBox = slide.shapes.add_textbox(Inches(1), Inches(5), Inches(12), Inches(1))
sFrame = sBox.text_frame
p2 = sFrame.paragraphs[0]
p2.text = "Mini Project Phase-1 Presentation\nPresented by: Sanket B. Majjagi"
p2.font.size = Pt(22)
p2.alignment = PP_ALIGN.CENTER

# ---------------------- CONTENT SLIDES -----------------------
slides_data = {
    "Introduction": [
        "Pneumonia is a lung infection that can be life-threatening if not diagnosed early.",
        "Chest X-rays are used for detection but manual interpretation is slow and subjective.",
        "Deep learning automates detection and increases diagnostic accuracy.",
        "MobileNetV2 is used for efficient real-time pneumonia detection."
    ],
    "Literature Survey": [
        "2018 – Paul Mooney (Kaggle Dataset): CNN achieved 84–87% accuracy.",
        "2019 – Rajpurkar et al. (CheXNet): DenseNet121 achieved radiologist-level accuracy.",
        "2020 – Kermany et al.: CNN model achieved around 90% accuracy.",
        "2023 – MobileNetV2 model reached 86–88% accuracy with lightweight design."
    ],
    "Existing System": [
        "Manual analysis by radiologists is accurate but time-consuming.",
        "Traditional CNNs (like VGG16, ResNet) require high computational power.",
        "Limited adaptability for low-resource clinical environments."
    ],
    "Proposed System / Methodology": [
        "Model: MobileNetV2 transfer learning approach for lightweight classification.",
        "Dataset: Kaggle Chest X-Ray dataset (Normal/Pneumonia).",
        "Dataset split: 80% training, 10% validation, 10% testing.",
        "Training includes class weighting to balance dataset and reduce bias.",
        "A Tkinter-based GUI for real-time pneumonia detection from images."
    ],
    "Expected Outcome": [
        "Accurate classification between NORMAL and PNEUMONIA chest X-rays.",
        "Achieved accuracy: 85–88% on test data.",
        "Lightweight, fast, and efficient model for clinical support.",
        "Interactive GUI for non-technical users."
    ],
    "Road Map": [
        "Phase 1 – Literature review and dataset setup ✔️",
        "Phase 2 – Model training and GUI development ✔️",
        "Phase 3 – Final testing and optimization 🔄",
        "Phase 4 – Report submission and project demonstration 🎯"
    ],
    "Conclusion": [
        "The project demonstrates the use of MobileNetV2 for pneumonia detection.",
        "Achieves high accuracy with reduced computational cost.",
        "The GUI improves usability and accessibility for medical practitioners.",
        "Future work: Multi-disease detection (e.g., TB, COVID-19)."
    ],
    "References": [
        "Paul Mooney, Chest X-Ray Images (Pneumonia) – Kaggle Dataset.",
        "Rajpurkar P. et al., CheXNet: Radiologist-Level Pneumonia Detection, 2017.",
        "Kermany D. et al., Image-Based Deep Learning, Cell, 2018.",
        "TensorFlow and Keras Documentation."
    ],
    "Demo & How to Use": [
        "Step 1: Run 'pneumonia_gui.py' in VS Code.",
        "Step 2: Click 'Select Chest X-ray Image' to upload an image.",
        "Step 3: View prediction — NORMAL ✅ or PNEUMONIA ⚠️ with likelihood.",
        "Add GUI screenshot on this slide before submission."
    ],
    "Thank You": [
        "Presented by: Sanket B. Majjagi",
        "Pneumonia Detection Using Deep Learning (MobileNetV2)",
        "Thank you for your attention!"
    ]
}

for title, points in slides_data.items():
    add_slide(title, points)

# ---------------------- SAVE PRESENTATION ---------------------
output_path = r"C:\Pneumonia_Project\Pneumonia_Detection_AI_Presentation_Final.pptx"
prs.save(output_path)
print(f"✅ Presentation generated successfully at:\n{output_path}")

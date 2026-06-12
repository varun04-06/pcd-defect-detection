
import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
import os

# Load model
model = YOLO("model/best.pt")

def get_severity(class_name, confidence, box_area, image_area):
    severity_map = {
        "short": 9, "open_circuit": 8, "missing_hole": 6,
        "mouse_bite": 5, "spur": 4, "spurious_copper": 3,
    }
    score = severity_map.get(class_name, 5)
    if confidence >= 0.75:   score = min(10, score + 1)
    elif confidence < 0.45:  score = max(1,  score - 1)
    area_ratio = box_area / image_area
    if area_ratio > 0.05:    score = min(10, score + 1); size_note = "large defect area"
    elif area_ratio < 0.005: score = max(1,  score - 1); size_note = "small defect area"
    else:                    size_note = "moderate defect area"
    if score >= 8:   level = "🔴 CRITICAL"
    elif score >= 5: level = "🟡 MODERATE"
    else:            level = "🟢 MINOR"
    root_cause = {
        "short":           "Solder bridging or copper etching failure",
        "open_circuit":    "Broken trace or incomplete etching",
        "missing_hole":    "Drill misalignment or drill bit failure",
        "mouse_bite":      "PCB edge routing or mechanical damage",
        "spur":            "Incomplete etching leaving copper spurs",
        "spurious_copper": "Etching resist failure or contamination",
    }
    reason = root_cause.get(class_name, "Unknown cause")
    return level, score, reason, size_note

def detect_defects(image):
    if image is None:
        return None, None, "⚠️ Please upload a PCB image."

    temp_path = "temp_input.jpg"
    cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    img_rgb = image.copy()
    h, w    = img_rgb.shape[:2]

    results = model.predict(source=temp_path, conf=0.45, iou=0.45, verbose=False)
    result  = results[0]

    # Detection image
    det_img = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)

    # Heatmap
    heatmap = np.zeros((h, w), dtype=np.float32)
    if len(result.boxes) > 0:
        for box in result.boxes:
            conf         = float(box.conf[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            cx,cy        = (x1+x2)//2, (y1+y2)//2
            bw,bh        = max(1,x2-x1), max(1,y2-y1)
            Y,X          = np.ogrid[:h,:w]
            sigma        = max(bw,bh) * 0.8
            gauss        = np.exp(-((X-cx)**2+(Y-cy)**2)/(2*sigma**2))
            heatmap     += gauss * conf
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    hm_color = cv2.applyColorMap((heatmap*255).astype(np.uint8), cv2.COLORMAP_JET)
    hm_rgb   = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    overlay  = cv2.addWeighted(img_rgb, 0.5, hm_rgb, 0.5, 0)

    # Build report
    report = "## 📊 Inspection Report\n\n"
    report += f"### Total Defects Found: {len(result.boxes)}\n\n"

    if len(result.boxes) == 0:
        report += "### ✅ Board Status: PASS\n"
        report += "No defects detected! Board passes inspection."
    else:
        critical = sum(1 for box in result.boxes
                      if get_severity(
                          model.names[int(box.cls[0])],
                          float(box.conf[0]),
                          ((int(box.xyxy[0][2])-int(box.xyxy[0][0])) *
                           (int(box.xyxy[0][3])-int(box.xyxy[0][1]))),
                          h*w)[0] == "🔴 CRITICAL")

        status = "❌ FAIL" if critical > 0 else "⚠️ REVIEW NEEDED"
        report += f"### Board Status: {status}\n\n"
        report += "---\n\n"

        for i, box in enumerate(result.boxes):
            cls_name     = model.names[int(box.cls[0])]
            conf         = float(box.conf[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            area         = (x2-x1) * (y2-y1)
            level, score, reason, size = get_severity(cls_name, conf, area, h*w)

            report += f"**#{i+1} — {cls_name.upper().replace(chr(95), ' ')}**\n"
            report += f"- Severity: {level} (Score: {score}/10)\n"
            report += f"- Confidence: {conf:.1%}\n"
            report += f"- Root Cause: {reason}\n"
            report += f"- Defect Size: {size}\n\n"

    return det_img, overlay, report

# ── Gradio UI ──
with gr.Blocks(theme=gr.themes.Soft(), title="PCB Defect Detection") as demo:
    gr.Markdown("""
    # 🔬 PCB Defect Detection System
    ### AI-Powered Quality Inspection | YOLOv8 + Severity Scoring + Explainability
    Upload a PCB image to automatically detect and classify defects.
    """)

    with gr.Row():
        input_img = gr.Image(label="📤 Upload PCB Image", type="numpy")

    with gr.Row():
        detect_btn = gr.Button("🔍 Analyze PCB", variant="primary", size="lg")

    with gr.Row():
        det_output  = gr.Image(label="🔍 Detection Result")
        heat_output = gr.Image(label="🌡️ Attention Heatmap")

    report_output = gr.Markdown(label="📊 Inspection Report")

    detect_btn.click(
        fn=detect_defects,
        inputs=input_img,
        outputs=[det_output, heat_output, report_output]
    )

    gr.Markdown("""
    ### Detectable Defect Types
    | Defect | Severity | Description |
    |--------|----------|-------------|
    | Short | 🔴 Critical | Solder bridging between traces |
    | Open Circuit | 🔴 Critical | Broken or incomplete trace |
    | Missing Hole | 🟡 Moderate | Drill misalignment |
    | Mouse Bite | 🟡 Moderate | Edge routing damage |
    | Spur | 🟢 Minor | Copper excess on trace |
    | Spurious Copper | 🟢 Minor | Unwanted copper deposits |
    """)

demo.launch()

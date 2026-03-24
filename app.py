"""
Semi-Supervised Medical Image Detection System — Main Streamlit Application.
Entry point: streamlit run app.py
"""

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

import config
from detector import load_detection_model, run_detection, generate_and_save_pseudo_labels
from detector.inference import parse_detections
from ui import apply_medical_theme, render_header, render_upload_section
from ui import render_analysis_panel, render_metrics_panel, render_view_toggle
from utils import load_image, compute_detection_metrics


# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Medical Detection System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_medical_theme()
render_header()

# -----------------------------------------------------------------------------
# Cached model loader
# -----------------------------------------------------------------------------
@st.cache_resource
def get_model():
    """Load YOLOv8 model once and reuse."""
    return load_detection_model()


# -----------------------------------------------------------------------------
# Sidebar: Upload
# -----------------------------------------------------------------------------
uploaded_file = render_upload_section()


# -----------------------------------------------------------------------------
# Main area
# -----------------------------------------------------------------------------
if uploaded_file is None:
    st.info("👈 Upload a medical image from the left panel to run AI detection.")
    st.markdown("---")
  
    st.stop()

# Load image
image_bytes = uploaded_file.read()
image_name = uploaded_file.name
pil_image = load_image(image_bytes)

# -----------------------------------------------------------------------------
# Detection Pipeline
# -----------------------------------------------------------------------------
with st.spinner("Running AI Analysis..."):
    model = get_model()

    # Run YOLO detection
    results, processing_time, annotated_image = run_detection(model, pil_image)

    all_detections = parse_detections(results)

    # Use all detections for display & metrics (avoids "no high-confidence" warning)
    detections = all_detections
    metrics = compute_detection_metrics(detections, processing_time)

    # Pseudo-labels only for high confidence (≥ 0.6) for semi-supervised simulation
    high_conf_detections = [
        d for d in all_detections
        if d["confidence"] >= config.PSEUDO_LABEL_CONFIDENCE_THRESHOLD
    ]
    pseudo_labels = generate_and_save_pseudo_labels(image_name, high_conf_detections)


# -----------------------------------------------------------------------------
# View toggle: Original | Detection
# -----------------------------------------------------------------------------
view_options = ["Original", "Detection"]
view = render_view_toggle(view_options, key="view_mode")

display_image = pil_image if view == "Original" else annotated_image

st.image(display_image, width="stretch", caption=f"View: {view}")


# -----------------------------------------------------------------------------
# Metrics Row
# -----------------------------------------------------------------------------
render_metrics_panel(
    total_objects=metrics["total_objects"],
    average_confidence=metrics["average_confidence"],
    processing_time_ms=metrics["processing_time_ms"],
)


# -----------------------------------------------------------------------------
# Right Panel: AI Analysis
# -----------------------------------------------------------------------------
render_analysis_panel(detections, processing_time, pseudo_labels)

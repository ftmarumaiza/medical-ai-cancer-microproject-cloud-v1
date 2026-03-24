"""
Reusable Streamlit components for the medical detection dashboard.
"""

from typing import List, Optional

import streamlit as st

import config


def render_header() -> None:
    """Top header: title and subtitle."""
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 0.25rem;'>"
        "AI Assisted Medical Detection System</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; color: #94a3b8; font-size: 1rem; margin-bottom: 2rem;'>"
        "Deep Learning Powered Clinical Decision Support</p>",
        unsafe_allow_html=True,
    )


def render_upload_section(key: str = "upload") -> Optional[any]:
    """Left panel: drag & drop image upload. Returns uploaded file or None."""
    with st.sidebar:
        st.markdown("### 📤 Upload Image")
        st.markdown("Drag & drop or browse for a medical image (e.g. colonoscopy).")
        uploaded = st.file_uploader(
            "Choose an image",
            type=["png", "jpg", "jpeg", "bmp"],
            key=key,
        )
    return uploaded


def _risk_level(confidence: float) -> tuple:
    if confidence >= config.RISK_HIGH_CONFIDENCE:  # ≥ 80%
        return "HIGH", "risk-high"
    if confidence >= config.RISK_MEDIUM_CONFIDENCE:
        return "MEDIUM", "risk-medium"
    return "LOW", "risk-low"


def render_analysis_panel(
    detections: List[dict],
    processing_time_seconds: float,
    pseudo_labels: List[dict],
) -> None:
    """
    Right panel: AI analysis - detection result, confidence, risk level.
    """
    with st.sidebar:
        st.markdown("### 🔬 AI Analysis")

        if not detections:
            st.info(
                "No objects detected in this image. The default model (YOLOv8 on COCO) recognizes everyday objects; "
                "for polyp/medical detection with high confidence, use a model fine-tuned on medical data (e.g. Kvasir)."
            )
            st.caption(f"Processing time: {processing_time_seconds*1000:.0f} ms")
            return

        top = detections[0]
        label = top.get("label", "object").capitalize()
        conf = top.get("confidence", 0.0)
        risk_name, risk_class = _risk_level(conf)

        st.markdown("**Detection Result:**")
        st.success(f"{label} Detected")

        st.markdown("**Confidence:**")
        st.metric("Confidence score", f"{conf*100:.1f}%", label_visibility="collapsed")

        st.markdown("**Risk Level:**")
        st.markdown(
            f'<p class="{risk_class}">{risk_name}</p>',
            unsafe_allow_html=True,
        )

        st.markdown("**Processing Time:**")
        st.caption(f"{processing_time_seconds*1000:.0f} ms")

        if pseudo_labels:
            st.markdown("---")
            st.markdown("**Pseudo-labels saved** (for semi-supervised pipeline):")
            for pl in pseudo_labels[:5]:
                st.caption(f"• {pl.get('label', '')} ({pl.get('confidence', 0)*100:.1f}%)")
            if len(pseudo_labels) > 5:
                st.caption(f"... and {len(pseudo_labels)-5} more")


def render_metrics_panel(
    total_objects: int,
    average_confidence: float,
    processing_time_ms: float,
) -> None:
    """Metrics row: total objects, avg confidence, processing time."""
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Objects Detected", total_objects)
    with c2:
        st.metric("Average Confidence", f"{average_confidence*100:.1f}%")
    with c3:
        st.metric("Processing Time", f"{processing_time_ms:.0f} ms")


def render_view_toggle(
    options: List[str],
    key: str = "view_mode",
) -> str:
    """Toggle buttons: Original / Detection / (optional Segmentation). Returns selected option."""
    st.markdown("**View:**")
    choice = st.radio(
        "View mode",
        options=options,
        key=key,
        horizontal=True,
        label_visibility="collapsed",
    )
    return choice

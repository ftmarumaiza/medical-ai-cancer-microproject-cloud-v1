"""
Dark medical dashboard theme: teal/cyan accents, soft shadows, rounded cards.
"""

MEDICAL_THEME_CSS = """
<style>
/* Base */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
}
[data-testid="stHeader"] { background: rgba(13, 17, 23, 0.95); }
.main .block-container { padding: 2rem 3rem 4rem; max-width: 1400px; }

/* Headers */
h1, h2, h3 {
    color: #e6edf3 !important;
    font-weight: 600;
}
h1 { font-size: 1.85rem; letter-spacing: -0.02em; }
h2 { font-size: 1.35rem; border-bottom: 1px solid #30363d; padding-bottom: 0.5rem; }

/* Cards */
[data-testid="stVerticalBlock"] > div {
    background: rgba(22, 27, 34, 0.85);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.25rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}
div[data-testid="stExpander"] {
    background: rgba(22, 27, 34, 0.9);
    border: 1px solid #30363d;
    border-radius: 12px;
}

/* Accent - Teal / Cyan */
.stMetric {
    background: linear-gradient(135deg, rgba(34, 211, 238, 0.12) 0%, rgba(20, 184, 166, 0.12) 100%);
    border: 1px solid rgba(34, 211, 238, 0.35);
    border-radius: 10px;
    padding: 1rem;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.2);
}
.stMetric label { color: #94a3b8 !important; font-size: 0.8rem !important; }
.stMetric [data-testid="stMetricValue"] { color: #22d3ee !important; font-weight: 700 !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0d9488 0%, #0891b2 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.25rem !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 10px rgba(13, 148, 136, 0.4) !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(34, 211, 238, 0.45) !important;
}

/* Upload area */
[data-testid="stFileUploader"] {
    background: rgba(22, 27, 34, 0.9);
    border: 2px dashed #30363d;
    border-radius: 12px;
    padding: 1.5rem;
}
[data-testid="stFileUploader"]:hover { border-color: #22d3ee; }
[data-testid="stFileUploader"] section { padding: 1rem !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] .stMarkdown { color: #94a3b8 !important; }

/* Risk badges */
.risk-high   { color: #f87171 !important; font-weight: 700; }
.risk-medium { color: #fbbf24 !important; font-weight: 700; }
.risk-low   { color: #34d399 !important; font-weight: 700; }

/* Spinner / loading */
.stSpinner > div { border-top-color: #22d3ee !important; }

/* Info / success */
.stSuccess { background: rgba(34, 211, 238, 0.15) !important; border-color: #22d3ee !important; }
</style>
"""


def apply_medical_theme() -> None:
    """Inject medical dashboard CSS into Streamlit."""
    import streamlit as st
    st.markdown(MEDICAL_THEME_CSS, unsafe_allow_html=True)

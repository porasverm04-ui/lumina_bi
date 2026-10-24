"""Shared white + orange look for every Lumina BI page.

Call inject_css() once, near the top of each page (after st.set_page_config).
Colors live in the variables below, so changing the accent is a one-line edit.
"""
import streamlit as st

ACCENT = "#FF6A00"
ACCENT_DARK = "#D95800"
TINT = "#FFF4EA"
INK = "#1A1A1A"
MUTED = "#555555"
RULE = "#F3D2B5"

# Orange-led palette for Plotly charts (orange + neutrals, readable when repeated)
PALETTE = [
    "#FF6A00", "#1A1A1A", "#FFB27A", "#8C3B00",
    "#FFD9B8", "#555555", "#D95800", "#F5A25D",
]


def inject_css():
    st.markdown(
        f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700;900&display=swap');

    :root {{
        --accent: {ACCENT};
        --accent-dark: {ACCENT_DARK};
        --tint: {TINT};
        --ink: {INK};
        --muted: {MUTED};
        --rule: {RULE};
    }}

    /* Base surfaces: white page, cream sidebar */
    [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
        background-color: #FFFFFF !important;
        color: var(--ink) !important;
    }}
    [data-testid="stSidebar"] {{
        background-color: var(--tint) !important;
        border-right: 1px solid var(--rule);
    }}
    .block-container {{ padding-top: 2rem !important; }}

    /* Classic serif headings */
    h1, h2, h3, h4 {{
        font-family: 'Playfair Display', Georgia, serif !important;
        color: var(--ink) !important;
        letter-spacing: -0.01em;
    }}
    h1 {{ border-bottom: 3px solid var(--accent); padding-bottom: 0.4rem; }}
    hr {{ border-color: var(--rule) !important; }}

    /* Buttons: orange outline, fill on hover; primary = solid orange */
    .stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {{
        background: #FFFFFF;
        color: var(--accent);
        border: 1.5px solid var(--accent);
        border-radius: 4px;
        font-weight: 600;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        background: var(--accent);
        color: #FFFFFF;
        border-color: var(--accent);
    }}
    .stButton > button[kind="primary"] {{
        background: var(--accent);
        color: #FFFFFF;
    }}
    .stButton > button[kind="primary"]:hover {{
        background: var(--accent-dark);
        border-color: var(--accent-dark);
    }}

    /* Alerts (success / info / warning): cream box, orange rule, dark text */
    [data-testid="stAlert"] > div {{
        background-color: var(--tint) !important;
        border-left: 4px solid var(--accent) !important;
        color: var(--ink) !important;
        border-radius: 2px;
    }}
    [data-testid="stAlert"] p {{ color: var(--ink) !important; }}

    /* Home-page hero + cards */
    .mega-title {{
        font-family: 'Playfair Display', Georgia, serif;
        font-weight: 900;
        font-size: clamp(3rem, 10vw, 8.5rem);
        letter-spacing: -0.04em;
        line-height: 0.9;
        text-transform: uppercase;
        margin-bottom: 3rem;
        color: var(--ink);
    }}
    .mega-title .dot {{ color: var(--accent); }}
    .sub-heading {{
        font-family: 'Playfair Display', Georgia, serif;
        font-weight: 700;
        font-size: clamp(1.6rem, 4vw, 3rem);
        letter-spacing: -0.02em;
        line-height: 1.05;
        text-transform: uppercase;
        margin: 2rem 0 1.5rem 0;
        color: var(--ink);
    }}
    .swiss-card {{
        border-top: 3px solid var(--accent);
        padding-top: 1rem;
        margin-bottom: 2rem;
    }}
    .swiss-card h4 {{
        font-size: 1.4rem;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
        color: var(--ink);
    }}
    .swiss-card p {{
        color: var(--muted);
        font-size: 0.95rem;
        line-height: 1.5;
    }}
</style>
""",
        unsafe_allow_html=True,
    )

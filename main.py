import streamlit as st
from theme import inject_css

# --- PAGE CONFIG ---
st.set_page_config(page_title="Lumina BI | Home", layout="wide")

# --- SHARED WHITE + ORANGE THEME ---
inject_css()

# --- HERO SECTION ---
# Using the exact phrasing style from your reference image
st.markdown('<div class="mega-title">LUMINA<br>BI<span class="dot">.</span></div>', unsafe_allow_html=True)

# --- ARCHITECTURE GRID ---
st.markdown('<div class="sub-heading">CARRY INNOVATION<br>WITH YOU</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="swiss-card">
        <h4>01 / Healing</h4>
        <p>LLM-driven contextual imputation and sequence pattern recognition. Zero manual cleanup required.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="swiss-card">
        <h4>02 / Workspace</h4>
        <p>Interactive multi-sheet data editor with real-time natural language query execution.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="swiss-card">
        <h4>03 / Export</h4>
        <p>Binary SQLite database push and comprehensive audit trails for absolute data governance.</p>
    </div>
    """, unsafe_allow_html=True)

# --- SYSTEM STATUS ---
st.markdown('<div class="swiss-card" style="margin-top: 4rem;"><h4>SYSTEM STATUS</h4></div>', unsafe_allow_html=True)

if "cleaned_df" in st.session_state:
    df = st.session_state["cleaned_df"]
    
    st.success("🟢 Active in Memory")
    st.dataframe(df.head(4), use_container_width=True)
else:
    st.info("⚪ System Idle. Navigate to the sidebar to initialize the pipeline.")
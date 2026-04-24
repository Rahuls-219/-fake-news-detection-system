"""
app.py
-------
Streamlit UI for the Fake News Detection System.
Supports Light & Dark mode, real-time prediction, confidence gauge, keyword highlights.

Run: streamlit run app.py
"""

import streamlit as st
import time
from utils import load_model, predict

# ──────────────────────────────────────────────
# PAGE CONFIG (must be first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# THEME DEFINITIONS
# ──────────────────────────────────────────────
LIGHT_THEME = {
    "bg": "#F8F9FA",
    "card": "#FFFFFF",
    "text": "#1A1A2E",
    "subtext": "#6C757D",
    "border": "#DEE2E6",
    "fake_bg": "#FFF0F0",
    "fake_border": "#FF4444",
    "fake_text": "#CC0000",
    "real_bg": "#F0FFF4",
    "real_border": "#00C851",
    "real_text": "#00703C",
    "badge_fake": "#FF4444",
    "badge_real": "#00C851",
    "accent": "#4361EE",
    "btn_text": "#FFFFFF",
    "input_bg": "#FFFFFF",
    "shadow": "rgba(0,0,0,0.08)",
    "tag_bg": "#EEF2FF",
    "tag_text": "#4361EE",
    "progress_bg": "#E9ECEF",
}

DARK_THEME = {
    "bg": "#0D1117",
    "card": "#161B22",
    "text": "#E6EDF3",
    "subtext": "#8B949E",
    "border": "#30363D",
    "fake_bg": "#1C0A0A",
    "fake_border": "#FF4444",
    "fake_text": "#FF6B6B",
    "real_bg": "#0A1C12",
    "real_border": "#00C851",
    "real_text": "#4CC97F",
    "badge_fake": "#FF4444",
    "badge_real": "#00C851",
    "accent": "#58A6FF",
    "btn_text": "#FFFFFF",
    "input_bg": "#0D1117",
    "shadow": "rgba(0,0,0,0.4)",
    "tag_bg": "#1C2333",
    "tag_text": "#58A6FF",
    "progress_bg": "#30363D",
}

# ──────────────────────────────────────────────
# SESSION STATE INIT
# ──────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "result" not in st.session_state:
    st.session_state.result = None
if "last_input" not in st.session_state:
    st.session_state.last_input = ""

T = DARK_THEME if st.session_state.dark_mode else LIGHT_THEME

# ──────────────────────────────────────────────
# MODEL LOADING (cached — loads only once)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model():
    try:
        model, vectorizer = load_model()
        return model, vectorizer, None
    except FileNotFoundError as e:
        return None, None, str(e)

model, vectorizer, model_error = get_model()

# ──────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────
st.markdown(f"""
<style>
  /* ── Page background ── */
  .stApp {{
    background-color: {T['bg']};
    color: {T['text']};
    font-family: 'Segoe UI', system-ui, sans-serif;
  }}

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header {{ visibility: hidden; }}
  .block-container {{ padding: 1.5rem 1rem 4rem; max-width: 800px; margin: 0 auto; }}

  /* ── Header bar ── */
  .header-bar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.5rem;
    padding: 0;
  }}
  .app-title {{
    font-size: 1.75rem;
    font-weight: 800;
    color: {T['text']};
    letter-spacing: -0.5px;
    line-height: 1.1;
  }}
  .app-subtitle {{
    font-size: 0.85rem;
    color: {T['subtext']};
    margin-top: 0.15rem;
  }}

  /* ── Card ── */
  .card {{
    background: {T['card']};
    border: 1px solid {T['border']};
    border-radius: 14px;
    padding: 1.5rem;
    box-shadow: 0 2px 12px {T['shadow']};
    margin-bottom: 1rem;
  }}

  /* ── Section label ── */
  .section-label {{
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: {T['subtext']};
    margin-bottom: 0.5rem;
  }}

  /* ── Result badge ── */
  .result-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 1.6rem;
    font-weight: 900;
    letter-spacing: 1px;
    padding: 0.3rem 1.2rem;
    border-radius: 50px;
    margin-bottom: 1rem;
  }}
  .badge-fake {{
    background: {T['fake_bg']};
    color: {T['fake_text']};
    border: 2px solid {T['fake_border']};
  }}
  .badge-real {{
    background: {T['real_bg']};
    color: {T['real_text']};
    border: 2px solid {T['real_border']};
  }}

  /* ── Confidence bar ── */
  .confidence-track {{
    background: {T['progress_bg']};
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
    margin-top: 0.3rem;
  }}
  .confidence-fill-fake {{
    height: 100%;
    background: linear-gradient(90deg, #FF7070, #FF4444);
    border-radius: 999px;
    transition: width 0.5s ease;
  }}
  .confidence-fill-real {{
    height: 100%;
    background: linear-gradient(90deg, #4CC97F, #00C851);
    border-radius: 999px;
    transition: width 0.5s ease;
  }}

  /* ── Keyword tag ── */
  .kw-tag {{
    display: inline-block;
    background: {T['tag_bg']};
    color: {T['tag_text']};
    border-radius: 6px;
    padding: 0.2rem 0.65rem;
    font-size: 0.82rem;
    font-weight: 600;
    margin-right: 0.4rem;
    margin-top: 0.3rem;
    font-family: 'Courier New', monospace;
  }}

  /* ── Insight box ── */
  .insight-box {{
    background: {T['tag_bg']};
    border-left: 4px solid {T['accent']};
    border-radius: 0 10px 10px 0;
    padding: 0.85rem 1rem;
    font-size: 0.9rem;
    color: {T['text']};
    line-height: 1.6;
    margin-top: 0.4rem;
  }}

  /* ── Divider ── */
  .result-divider {{
    border: none;
    border-top: 1px solid {T['border']};
    margin: 1rem 0;
  }}

  /* ── Prob row ── */
  .prob-row {{
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    color: {T['subtext']};
    margin-bottom: 0.2rem;
  }}

  /* ── Streamlit textarea override ── */
  .stTextArea textarea {{
    background-color: {T['input_bg']} !important;
    color: {T['text']} !important;
    border: 1.5px solid {T['border']} !important;
    border-radius: 10px !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
  }}
  .stTextArea textarea:focus {{
    border-color: {T['accent']} !important;
    box-shadow: 0 0 0 3px rgba(67, 97, 238, 0.15) !important;
  }}

  /* ── Buttons ── */
  .stButton > button {{
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    height: 44px !important;
    transition: all 0.2s ease !important;
  }}
  .stButton > button:first-child {{
    background: linear-gradient(135deg, {T['accent']}, #7B5EA7) !important;
    color: white !important;
    border: none !important;
  }}
  .stButton > button:hover {{
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px {T['shadow']} !important;
  }}

  /* ── Spinner ── */
  .analyzing-box {{
    text-align: center;
    padding: 1.5rem;
    color: {T['subtext']};
    font-size: 0.95rem;
    font-style: italic;
  }}
  .analyzing-box .spinner {{
    font-size: 1.8rem;
    animation: spin 1.2s linear infinite;
    display: inline-block;
  }}
  @keyframes spin {{
    from {{ transform: rotate(0deg); }}
    to {{ transform: rotate(360deg); }}
  }}

  /* ── Example section ── */
  .examples-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
    margin-top: 0.5rem;
  }}
  .example-card {{
    background: {T['card']};
    border: 1px solid {T['border']};
    border-radius: 10px;
    padding: 0.75rem;
    font-size: 0.82rem;
    color: {T['subtext']};
    cursor: pointer;
    transition: border-color 0.2s;
  }}
  .example-card:hover {{
    border-color: {T['accent']};
    color: {T['text']};
  }}
  .example-card .ex-label {{
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.3rem;
  }}
  .ex-fake {{ color: {T['fake_text']}; }}
  .ex-real {{ color: {T['real_text']}; }}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
col_title, col_toggle = st.columns([4, 1])
with col_title:
    st.markdown("""
    <div class="header-bar">
      <div>
        <div class="app-title">🔍 Fake News Detector</div>
        <div class="app-subtitle">Real-Time · Explainable · Instant Verification</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
with col_toggle:
    st.write("")
    toggle_label = "☀️ Light" if st.session_state.dark_mode else "🌙 Dark"
    if st.button(toggle_label, key="theme_toggle"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.session_state.result = None
        st.rerun()

st.markdown("<hr style='border:none;border-top:1px solid " + T['border'] + ";margin:0.5rem 0 1.2rem'>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# MODEL ERROR GATE
# ──────────────────────────────────────────────
if model_error:
    st.error(f"""
    **Model not found!**  
    Run these commands first:
    ```bash
    python generate_dataset.py
    python train_model.py
    ```
    Then restart: `streamlit run app.py`
    """)
    st.stop()

# ──────────────────────────────────────────────
# EXAMPLE INPUTS
# ──────────────────────────────────────────────
EXAMPLES = [
    {
        "label": "FAKE",
        "text": "SHOCKING: Scientists discover miracle cure for all diseases hidden by Big Pharma for decades. Share before deleted!"
    },
    {
        "label": "FAKE",
        "text": "BREAKING: Government secretly putting mind control chemicals in drinking water, whistleblower reveals truth!"
    },
    {
        "label": "REAL",
        "text": "Federal Reserve raises interest rates by 25 basis points amid ongoing inflation concerns, officials confirmed."
    },
    {
        "label": "REAL",
        "text": "University researchers develop new battery technology that charges 40 percent faster, study published in Nature."
    },
]

# ──────────────────────────────────────────────
# INPUT SECTION
# ──────────────────────────────────────────────
st.markdown('<div class="section-label">📝 Enter News Article / Headline</div>', unsafe_allow_html=True)

news_input = st.text_area(
    label="news_input",
    label_visibility="collapsed",
    height=150,
    placeholder="Paste a news headline or article text here…\n\nExample: 'Scientists discover miracle cure hidden by Big Pharma…'",
    value=st.session_state.last_input,
    key="news_text",
)

col_check, col_clear = st.columns([3, 1])
with col_check:
    check_btn = st.button("🔍 Check News", use_container_width=True, key="check")
with col_clear:
    if st.button("🗑️ Clear", use_container_width=True, key="clear"):
        st.session_state.result = None
        st.session_state.last_input = ""
        st.rerun()

# ──────────────────────────────────────────────
# PREDICTION LOGIC
# ──────────────────────────────────────────────
if check_btn:
    if not news_input.strip():
        st.warning("⚠️ Please enter some text before checking.")
    else:
        st.session_state.last_input = news_input
        result_placeholder = st.empty()
        result_placeholder.markdown("""
        <div class="card analyzing-box">
          <div class="spinner">⏳</div><br>
          Analyzing news… <br>
          <small>Extracting patterns · Running classifier</small>
        </div>
        """, unsafe_allow_html=True)

        t0 = time.time()
        result = predict(news_input, model, vectorizer)
        elapsed = (time.time() - t0) * 1000

        result["elapsed_ms"] = round(elapsed, 1)
        st.session_state.result = result
        result_placeholder.empty()
        st.rerun()

# ──────────────────────────────────────────────
# RESULTS DISPLAY
# ──────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result
    is_fake = r["label"] == "FAKE"
    badge_class = "badge-fake" if is_fake else "badge-real"
    icon = "🚨" if is_fake else "✅"
    fill_class = "confidence-fill-fake" if is_fake else "confidence-fill-real"

    st.markdown('<div class="card">', unsafe_allow_html=True)

    # ── Verdict ──
    st.markdown(f"""
    <div class="result-badge {badge_class}">
      {icon} {r['label']}
    </div>
    """, unsafe_allow_html=True)

    # ── Confidence bar ──
    st.markdown(f"""
    <div class="section-label">Confidence Score</div>
    <div class="prob-row">
      <span>🚨 Fake: <b>{r['fake_prob']}%</b></span>
      <span>✅ Real: <b>{r['real_prob']}%</b></span>
    </div>
    <div class="confidence-track">
      <div class="{fill_class}" style="width:{r['confidence']}%"></div>
    </div>
    <div style="text-align:right;font-size:0.8rem;color:{T['subtext']};margin-top:0.3rem;">
      {r['confidence']}% confidence · {r['elapsed_ms']}ms
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="result-divider">', unsafe_allow_html=True)

    # ── Keywords ──
    st.markdown('<div class="section-label">🔑 Top Signal Keywords</div>', unsafe_allow_html=True)
    kw_html = "".join(f'<span class="kw-tag">{kw}</span>' for kw in r["top_keywords"])
    st.markdown(kw_html, unsafe_allow_html=True)

    st.markdown('<hr class="result-divider">', unsafe_allow_html=True)

    # ── Insight ──
    st.markdown('<div class="section-label">💡 Insight</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="insight-box">{r["insight"]}</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# EXAMPLE INPUTS SECTION
# ──────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">🧪 Try These Examples</div>', unsafe_allow_html=True)

cols = st.columns(2)
for i, ex in enumerate(EXAMPLES):
    with cols[i % 2]:
        label_class = "ex-fake" if ex["label"] == "FAKE" else "ex-real"
        label_icon = "🚨" if ex["label"] == "FAKE" else "✅"
        if st.button(
            f"{label_icon} {ex['label']}: {ex['text'][:60]}…",
            key=f"ex_{i}",
            use_container_width=True
        ):
            st.session_state.last_input = ex["text"]
            result = predict(ex["text"], model, vectorizer)
            result["elapsed_ms"] = 0.0
            st.session_state.result = result
            st.rerun()

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;margin-top:2.5rem;font-size:0.78rem;color:{T['subtext']}">
  Built with ❤️ by <b>CodeTrio</b> · FusionX Hackathon 2026 · Presidency University, Bengaluru<br>
  Powered by <b>Logistic Regression + TF-IDF</b> · Runs fully offline · No external APIs
</div>
""", unsafe_allow_html=True)

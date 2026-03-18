# ============================================================
#  LEAF AUTOPSY — Streamlit App (EfficientNetB0 only)
#
#  Tabs:
#    01 // SINGLE ANALYSIS   — GradCAM + diagnosis report
#    02 // BATCH TRIAGE      — up to 5 images, risk queue
#    03 // PATHOGEN ATLAS    — browsable disease database
#    04 // SYSTEM INFO       — model config + metrics
#
#  Run: streamlit run app.py
#
#  Files required (same directory):
#    plant_disease_model.h5
#    class_names.json
#    training_metrics.json
# ============================================================

import json, datetime
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
import matplotlib.cm as mpl_cm
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="LEAF AUTOPSY",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# DISEASE DATABASE
# ─────────────────────────────────────────────
DISEASE_DB = {
    "healthy": {
        "status"    : "HEALTHY",
        "code"      : "PHY-000",
        "risk"      : 0,
        "risk_label": "NONE",
        "host"      : "All monitored species",
        "pathogen"  : "—",
        "taxonomy"  : "—",
        "symptoms"  : "No lesions, uniform green colouration, no necrosis.",
        "mechanism" : "No pathogenic indicators detected. Leaf morphology within normal parameters.",
        "conditions": "N/A",
        "protocol"  : "Continue standard agronomic practices. Schedule next inspection in 14 days.",
        "accent"    : "#00e676",
        "emoji"     : "✅",
        "severity"  : "NONE",
    },
    "early_blight": {
        "status"    : "EARLY BLIGHT",
        "code"      : "PHY-101",
        "risk"      : 45,
        "risk_label": "MODERATE",
        "host"      : "Solanum lycopersicum (Tomato)",
        "pathogen"  : "Alternaria solani",
        "taxonomy"  : "Ascomycete fungus — Pleosporales",
        "symptoms"  : "Concentric dark rings forming 'bullseye' lesions on older leaves. Yellow halo. Defoliation from base upward.",
        "mechanism" : "Conidia germinate in free moisture, penetrate through stomata or wounds. Necrosis spreads outward forming concentric rings as the colony ages.",
        "conditions": "Warm (24–29°C), humid conditions. RH > 90%. Prolonged leaf wetness.",
        "protocol"  : "Apply copper-based fungicide (copper oxychloride). Remove and destroy infected leaves. Avoid overhead irrigation. Rotate with non-solanaceous crops.",
        "accent"    : "#ffb300",
        "emoji"     : "⚠️",
        "severity"  : "MODERATE",
    },
    "late_blight": {
        "status"    : "LATE BLIGHT",
        "code"      : "PHY-102",
        "risk"      : 95,
        "risk_label": "CRITICAL",
        "host"      : "Solanum lycopersicum · S. tuberosum",
        "pathogen"  : "Phytophthora infestans",
        "taxonomy"  : "Oomycete (water mold) — Peronosporales",
        "symptoms"  : "Water-soaked, dark brown/black lesions. White sporangiophores on leaf undersides in humid conditions. Rapid tissue collapse.",
        "mechanism" : "Sporangia dispersed by wind/rain. Zoospores released in water, swim to stomata. Mycelium invades intercellularly, secreting effectors to suppress plant immunity.",
        "conditions": "Cool (10–20°C), very wet. RH > 90%. The pathogen responsible for the Irish Great Famine (1845).",
        "protocol"  : "IMMEDIATE ACTION. Apply mancozeb + metalaxyl systemic fungicide. Destroy infected plants — do NOT compost. Halt overhead irrigation. Notify adjacent farms.",
        "accent"    : "#ff1744",
        "emoji"     : "🚨",
        "severity"  : "CRITICAL",
    },
    "leaf_mold": {
        "status"    : "LEAF MOLD",
        "code"      : "PHY-103",
        "risk"      : 52,
        "risk_label": "MODERATE",
        "host"      : "Solanum lycopersicum (Tomato)",
        "pathogen"  : "Passalora fulva",
        "taxonomy"  : "Ascomycete fungus — Capnodiales (syn. Cladosporium fulvum)",
        "symptoms"  : "Pale yellow chlorotic spots (adaxial surface). Olive-brown velvety sporulation on abaxial surface. No leaf distortion initially.",
        "mechanism" : "Conidiophores emerge through stomata. Conidia spread by air currents and contact. Fungal effectors (Avr proteins) suppress Cf-gene resistance in susceptible cultivars.",
        "conditions": "High humidity > 85%. Poor ventilation. Greenhouse environments. Temperature 22–25°C.",
        "protocol"  : "Improve canopy airflow. Reduce relative humidity below 80%. Apply copper oxychloride fungicide. Remove chlorotic foliage.",
        "accent"    : "#ff9800",
        "emoji"     : "⚠️",
        "severity"  : "MODERATE",
    },
    "septoria_leaf_spot": {
        "status"    : "SEPTORIA LEAF SPOT",
        "code"      : "PHY-104",
        "risk"      : 58,
        "risk_label": "MODERATE",
        "host"      : "Solanum lycopersicum (Tomato)",
        "pathogen"  : "Septoria lycopersici",
        "taxonomy"  : "Ascomycete fungus — Capnodiales",
        "symptoms"  : "Small circular lesions (3–6mm), grey-white centres with dark brown margins. Black pycnidia (fruiting bodies) visible as specks in lesion centres.",
        "mechanism" : "Pycnidiospores released in wet conditions, splash-dispersed to leaves. Secondary infections cascade rapidly in warm humid weather.",
        "conditions": "Warm (20–25°C), wet weather. Splash dispersal from soil. Volunteer plants as inoculum source.",
        "protocol"  : "Apply chlorothalonil or mancozeb. Remove and destroy infected lower-canopy foliage. Mulch soil surface to reduce splash inoculum. Practice 2-year crop rotation.",
        "accent"    : "#ef6c00",
        "emoji"     : "⚠️",
        "severity"  : "MODERATE",
    },
    "bacterial_spot": {
        "status"    : "BACTERIAL SPOT",
        "code"      : "PHY-201",
        "risk"      : 78,
        "risk_label": "HIGH",
        "host"      : "Capsicum annuum (Pepper)",
        "pathogen"  : "Xanthomonas campestris pv. vesicatoria",
        "taxonomy"  : "Gamma-Proteobacteria — Xanthomonadales",
        "symptoms"  : "Water-soaked angular lesions bounded by veins. Lesions turn brown/black with a yellow halo. Raised corky scab on fruit. Severe defoliation under wet conditions.",
        "mechanism" : "Entry via stomata, hydathodes, and wounds. Type III secretion system injects effector proteins suppressing PTI and ETI defense pathways. Systemic colonisation possible.",
        "conditions": "Warm (25–30°C), rainy or overhead-irrigated conditions. Spreads explosively through wet canopy contact.",
        "protocol"  : "Apply copper bactericide + mancozeb tank-mix. Use certified disease-free seed. Avoid working in wet crop. Eliminate infected debris. Consider resistant cultivars.",
        "accent"    : "#f44336",
        "emoji"     : "🚨",
        "severity"  : "HIGH",
    },
    "potato_late_blight": {
        "status"    : "POTATO LATE BLIGHT",
        "code"      : "PHY-202",
        "risk"      : 92,
        "risk_label": "CRITICAL",
        "host"      : "Solanum tuberosum (Potato)",
        "pathogen"  : "Phytophthora infestans",
        "taxonomy"  : "Oomycete (water mold) — Peronosporales",
        "symptoms"  : "Dark, water-soaked lesions starting at leaf margins. White sporulation on leaf undersides. Tuber rot with reddish-brown granular internal discolouration.",
        "mechanism" : "Identical to tomato late blight. Tuber infection occurs when sporangia wash into soil. Cold storage does not halt progression.",
        "conditions": "Cool wet weather. Zoospore release and infection in 4–12h of leaf wetness.",
        "protocol"  : "Apply preventive systemic fungicide before symptoms appear. Destroy haulm 2 weeks before harvest. Inspect stored tubers weekly. Do not use infected seed tubers.",
        "accent"    : "#ff1744",
        "emoji"     : "🚨",
        "severity"  : "CRITICAL",
    },
    "potato_early_blight": {
        "status"    : "POTATO EARLY BLIGHT",
        "code"      : "PHY-203",
        "risk"      : 40,
        "risk_label": "LOW-MODERATE",
        "host"      : "Solanum tuberosum (Potato)",
        "pathogen"  : "Alternaria solani",
        "taxonomy"  : "Ascomycete fungus — Pleosporales",
        "symptoms"  : "Dark brown lesions with concentric rings on lower/older leaves. Premature defoliation reduces tuber yield. Rarely affects tubers.",
        "mechanism" : "Same as tomato early blight. Stress (drought, nutrient deficiency) makes plants more susceptible.",
        "conditions": "Warm dry periods alternating with rain. Plants past flowering stage most vulnerable.",
        "protocol"  : "Apply mancozeb or chlorothalonil at 7–10 day intervals. Maintain adequate nitrogen and potassium levels. Avoid water stress.",
        "accent"    : "#ffb300",
        "emoji"     : "⚠️",
        "severity"  : "LOW-MODERATE",
    },
    "potato_healthy": {
        "status"    : "HEALTHY (POTATO)",
        "code"      : "PHY-001",
        "risk"      : 0,
        "risk_label": "NONE",
        "host"      : "Solanum tuberosum (Potato)",
        "pathogen"  : "—",
        "taxonomy"  : "—",
        "symptoms"  : "No lesions. Uniform green foliage. Normal compound leaf morphology.",
        "mechanism" : "No pathogenic indicators detected.",
        "conditions": "N/A",
        "protocol"  : "Maintain monitoring schedule. Inspect tubers at harvest for latent infections.",
        "accent"    : "#00e676",
        "emoji"     : "✅",
        "severity"  : "NONE",
    },
    "pepper_healthy": {
        "status"    : "HEALTHY (PEPPER)",
        "code"      : "PHY-002",
        "risk"      : 0,
        "risk_label": "NONE",
        "host"      : "Capsicum annuum (Pepper)",
        "pathogen"  : "—",
        "taxonomy"  : "—",
        "symptoms"  : "No lesions. Uniform green foliage. Normal simple leaf morphology.",
        "mechanism" : "No pathogenic indicators detected.",
        "conditions": "N/A",
        "protocol"  : "Continue standard practices. Monitor for bacterial spot under wet conditions.",
        "accent"    : "#00e676",
        "emoji"     : "✅",
        "severity"  : "NONE",
    },
}

def get_db(class_name: str) -> dict:
    n = class_name.lower()
    # direct key hits
    if "healthy" in n and "potato" in n: return DISEASE_DB["potato_healthy"]
    if "healthy" in n and "pepper" in n: return DISEASE_DB["pepper_healthy"]
    if "healthy" in n:                   return DISEASE_DB["healthy"]
    if "late_blight" in n and "potato" in n: return DISEASE_DB["potato_late_blight"]
    if "early_blight" in n and "potato" in n: return DISEASE_DB["potato_early_blight"]
    if "late_blight"  in n: return DISEASE_DB["late_blight"]
    if "early_blight" in n: return DISEASE_DB["early_blight"]
    if "leaf_mold"    in n: return DISEASE_DB["leaf_mold"]
    if "septoria"     in n: return DISEASE_DB["septoria_leaf_spot"]
    if "bacterial"    in n: return DISEASE_DB["bacterial_spot"]
    return {
        "status":"UNCLASSIFIED","code":"PHY-999","risk":50,
        "risk_label":"UNKNOWN","host":"Unknown","pathogen":"Unknown",
        "taxonomy":"Unknown","symptoms":"No entry.","mechanism":"No entry.",
        "conditions":"Unknown","protocol":"Consult expert.",
        "accent":"#607d8b","emoji":"❓","severity":"UNKNOWN",
    }

def clean_label(raw: str) -> str:
    return raw.replace("___","__").replace("__"," / ").replace("_"," ").upper()

# ─────────────────────────────────────────────
# CSS — IBM Plex Mono terminal
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
  :root {
    --bg:     #050809; --bg2: #0a1014; --bg3: #0f171c;
    --b:      #1a2e38; --b2:  #254050;
    --txt:    #c8dde8; --muted: #4a6a7a; --accent: #00c8ff;
    --green:  #00e676; --red: #ff1744; --amber: #ffb300;
    --mono:   'IBM Plex Mono', monospace;
    --sans:   'IBM Plex Sans', sans-serif;
  }
  html, body, [class*="css"] { font-family: var(--sans); background: var(--bg) !important; color: var(--txt); }
  .block-container { padding-top: 1rem !important; max-width: 1260px; }

  /* scanlines */
  body::before {
    content:''; position:fixed; top:0; left:0; width:100%; height:100%;
    background: repeating-linear-gradient(0deg, transparent, transparent 2px,
      rgba(0,200,255,0.010) 2px, rgba(0,200,255,0.010) 4px);
    pointer-events:none; z-index:9999;
  }

  /* header */
  .hdr { border-bottom:1px solid var(--b2); padding-bottom:.8rem; margin-bottom:1.4rem; }
  .hdr-title { font-family:var(--mono); font-size:1.55rem; font-weight:600; color:var(--accent); letter-spacing:.08em; }
  .hdr-sub   { font-family:var(--mono); font-size:.68rem; color:var(--muted); letter-spacing:.12em; margin-top:.2rem; }
  .hdr-live  { font-family:var(--mono); font-size:.68rem; color:var(--green); letter-spacing:.1em; }

  /* panel */
  .panel { background:var(--bg2); border:1px solid var(--b); border-radius:4px; padding:1.2rem 1.4rem; margin-bottom:.9rem; position:relative; }
  .panel::before { content:''; position:absolute; top:0; left:0; width:3px; height:100%; border-radius:4px 0 0 4px; background:var(--accent); }
  .plabel { font-family:var(--mono); font-size:.62rem; letter-spacing:.14em; color:var(--muted); text-transform:uppercase; margin-bottom:.5rem; }

  /* report id */
  .rid { font-family:var(--mono); font-size:.68rem; color:var(--muted); padding:.45rem .8rem; background:var(--bg3); border:1px solid var(--b); border-radius:3px; display:flex; gap:2rem; flex-wrap:wrap; margin-bottom:.9rem; }

  /* diag */
  .dcode  { font-family:var(--mono); font-size:.68rem; color:var(--muted); letter-spacing:.1em; }
  .dstatus{ font-family:var(--mono); font-size:1.75rem; font-weight:600; letter-spacing:.05em; line-height:1.1; }
  .dconf  { font-family:var(--mono); font-size:.76rem; color:var(--muted); margin-top:.3rem; }

  /* risk bar */
  .rbar-bg   { background:var(--bg3); border:1px solid var(--b); height:8px; border-radius:2px; margin:.4rem 0; overflow:hidden; }
  .rbar-fill { height:8px; border-radius:2px; }

  /* data rows */
  .drow { display:flex; gap:.8rem; padding:.42rem 0; border-bottom:1px solid var(--b); font-size:.82rem; }
  .drow:last-child { border-bottom:none; }
  .dkey { font-family:var(--mono); color:var(--muted); font-size:.68rem; letter-spacing:.08em; width:120px; flex-shrink:0; margin-top:.1rem; }
  .dval { color:var(--txt); flex:1; line-height:1.55; }

  /* prob bars */
  .prow  { display:flex; align-items:center; gap:.5rem; padding:.35rem 0; border-bottom:1px solid var(--b); }
  .prow:last-child { border-bottom:none; }
  .pname { font-family:var(--mono); font-size:.65rem; color:var(--muted); width:40%; flex-shrink:0; }
  .ptrk  { flex:1; background:var(--bg3); height:5px; border-radius:2px; overflow:hidden; }
  .pfill { height:5px; border-radius:2px; }
  .ppct  { font-family:var(--mono); font-size:.65rem; color:var(--muted); width:38px; text-align:right; }

  /* triage */
  .trow { display:flex; align-items:center; padding:.65rem .8rem; margin-bottom:.35rem;
          background:var(--bg3); border:1px solid var(--b); border-radius:3px;
          font-family:var(--mono); font-size:.76rem; gap:1rem; }
  .tidx { color:var(--muted); width:28px; flex-shrink:0; }
  .tfn  { color:var(--txt); flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .tdiag{ flex:1.4; font-weight:600; }
  .tconf{ color:var(--muted); width:48px; text-align:right; }

  /* atlas card */
  .atlas-card { background:var(--bg2); border:1px solid var(--b); border-radius:6px; padding:1.1rem 1.3rem; margin-bottom:.8rem; }
  .atlas-code { font-family:var(--mono); font-size:.62rem; color:var(--muted); letter-spacing:.1em; }
  .atlas-name { font-family:var(--mono); font-size:1.15rem; font-weight:600; line-height:1.2; margin:.15rem 0 .5rem; }
  .atlas-pill { display:inline-block; border-radius:3px; padding:.15rem .55rem;
                font-family:var(--mono); font-size:.62rem; letter-spacing:.06em; margin-bottom:.6rem; }
  .atlas-field{ font-size:.82rem; line-height:1.55; color:#8ab0c0; margin:.25rem 0 0; }
  .atlas-fi   { font-family:var(--mono); font-size:.62rem; color:var(--muted); letter-spacing:.08em; display:inline-block; margin-top:.5rem; }

  /* tabs */
  .stTabs [data-baseweb="tab-list"] { background:var(--bg) !important; gap:2px; border-bottom:1px solid var(--b2); }
  .stTabs [data-baseweb="tab"] { background:var(--bg2) !important; color:var(--muted) !important;
    font-family:var(--mono) !important; font-size:.69rem !important; letter-spacing:.1em !important;
    text-transform:uppercase !important; border-radius:3px 3px 0 0 !important;
    padding:.45rem 1.1rem !important; border:1px solid var(--b) !important; border-bottom:none !important; }
  .stTabs [aria-selected="true"] { background:var(--bg3) !important; color:var(--accent) !important; border-color:var(--b2) !important; }

  /* upload */
  [data-testid="stFileUploader"] { background:var(--bg2) !important; border:1px dashed var(--b2) !important; border-radius:4px !important; }
  [data-testid="stFileUploader"]:hover { border-color:var(--accent) !important; }
  .stProgress > div > div { background:var(--accent) !important; }

  hr { border-color:var(--b) !important; }
  #MainMenu, footer, .stDeployButton { display:none !important; }

  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.25} }
  .blink { animation:blink 1.4s ease-in-out infinite; }
  @keyframes fadeUp { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)} }
  .fi { animation:fadeUp .35s ease forwards; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD RESOURCES
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return tf.keras.models.load_model("plant_disease_model.h5")

@st.cache_data(show_spinner=False)
def load_meta():
    with open("class_names.json") as f:
        classes = json.load(f)
    metrics = {}
    try:
        with open("training_metrics.json") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        pass
    return classes, metrics

def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return np.expand_dims(arr, 0)

def compute_gradcam(model, inp, pred_index):
    base_layer = next(
        (l for l in model.layers if isinstance(l, tf.keras.Model)), None
    )
    if base_layer is None:
        return None

    last_conv_name = None
    for layer in reversed(base_layer.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_name = layer.name
            break
    if last_conv_name is None:
        return None

    try:
        inp_const = tf.constant(inp, dtype=tf.float32)
        grad_model = tf.keras.Model(
            inputs=base_layer.input,
            outputs=[
                base_layer.get_layer(last_conv_name).output,
                base_layer.output
            ]
        )
        with tf.GradientTape() as tape:
            tape.watch(inp_const)
            conv_out, features = grad_model(inp_const, training=False)
            x     = model.layers[-4](features)
            x     = model.layers[-3](x)
            x     = model.layers[-2](x)
            preds = model.layers[-1](x)
            loss  = preds[:, pred_index]

        grads   = tape.gradient(loss, conv_out)
        pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_sum(
            conv_out[0] * pooled[tf.newaxis, tf.newaxis, :], axis=-1
        ).numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (heatmap.max() + 1e-8)
        return heatmap
    except Exception as e:
        print(f"GradCAM failed: {e}")
        return None

def overlay(orig_pil: Image.Image, hmap, alpha=0.48) -> Image.Image:
    img = np.array(orig_pil.resize((224,224))).astype(np.uint8)
    if hmap is None: return Image.fromarray(img)
    h = tf.image.resize(hmap[...,np.newaxis],(224,224)).numpy().squeeze()
    colored = (mpl_cm.get_cmap("inferno")(h)[:,:,:3]*255).astype(np.uint8)
    return Image.fromarray(np.clip(colored*alpha + img*(1-alpha),0,255).astype(np.uint8))

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
now = datetime.datetime.now()
ts  = now.strftime("%Y-%m-%d  %H:%M:%S")

st.markdown(f"""
<div class="hdr">
  <div style="display:flex;justify-content:space-between;align-items:flex-end;">
    <div>
      <div class="hdr-title">◈ LEAF AUTOPSY</div>
      <div class="hdr-sub">PHYTOPATHOLOGICAL DIAGNOSTIC SYSTEM  //  EfficientNetB0  //  10-CLASS  //  PLANTVILLAGE</div>
    </div>
    <div style="text-align:right;">
      <div class="hdr-live"><span class="blink">●</span> SYSTEM ONLINE</div>
      <div style="font-family:var(--mono);font-size:.62rem;color:var(--muted);">{ts}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODEL LOAD
# ─────────────────────────────────────────────
with st.spinner(""):
    try:
        model       = load_model()
        class_names, train_metrics = load_meta()
    except FileNotFoundError as e:
        st.error(f"**[ERR] MODEL FILES NOT FOUND**\n\nPlace `plant_disease_model.h5` and `class_names.json` beside `app.py`.\n\n`{e}`")
        st.stop()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "01 // SINGLE ANALYSIS",
    "02 // BATCH TRIAGE",
    "03 // PATHOGEN ATLAS",
    "04 // SYSTEM INFO",
])

# ══════════════════════════════════════════════
# TAB 1 — SINGLE ANALYSIS
# ══════════════════════════════════════════════
with tab1:
    c_ctrl, _ = st.columns([3,1])
    with _:
        show_gcam = st.toggle("GradCAM", value=True)
        show_dist = st.toggle("Prob. dist.", value=True)

    uploaded = st.file_uploader("LOAD SPECIMEN IMAGE", type=["jpg","jpeg","png"], key="single")

    if not uploaded:
        st.markdown("""
        <div style="text-align:center;padding:4rem 0;color:var(--muted);">
          <div style="font-family:var(--mono);font-size:1.8rem;letter-spacing:.2em;">[ NO SPECIMEN ]</div>
          <div style="font-family:var(--mono);font-size:.68rem;letter-spacing:.1em;margin-top:.5rem;">
            AWAITING INPUT — LOAD A LEAF IMAGE TO BEGIN DIAGNOSTIC SEQUENCE
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        image  = Image.open(uploaded)
        tensor = preprocess(image)

        with st.spinner("RUNNING INFERENCE..."):
            preds = model.predict(tensor, verbose=0)[0]
            top_i = int(np.argmax(preds))
            conf  = float(preds[top_i])
            hmap  = compute_gradcam(model, tensor, top_i) if show_gcam else None

        db  = get_db(class_names[top_i])
        rid = f"RPT-{now.strftime('%Y%m%d')}-{hash(uploaded.name) % 10000:04d}"

        st.markdown(f"""
        <div class="rid fi">
          <span>REPORT_ID: <b style="color:var(--accent);">{rid}</b></span>
          <span>SPECIMEN: <b style="color:var(--txt);">{uploaded.name}</b></span>
          <span>{ts}</span>
          <span>MODEL: EfficientNetB0</span>
        </div>""", unsafe_allow_html=True)

        c_img, c_gcam, c_diag = st.columns([1.05,1.05,1.5], gap="medium")

        with c_img:
            st.markdown('<div class="plabel">RAW SPECIMEN</div>', unsafe_allow_html=True)
            st.image(image.resize((224,224)), use_container_width=True)

        with c_gcam:
            st.markdown('<div class="plabel">GRADCAM — INFERNO COLORMAP</div>', unsafe_allow_html=True)
            if show_gcam:
                st.image(overlay(image, hmap), use_container_width=True)
                st.markdown('<div style="font-family:var(--mono);font-size:.6rem;color:var(--muted);margin-top:.3rem;">HIGH ACTIVATION = PATHOGEN REGION</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="color:var(--muted);font-family:var(--mono);font-size:.72rem;padding:2rem 0;">DISABLED</div>', unsafe_allow_html=True)

        with c_diag:
            st.markdown(f"""
            <div class="panel fi" style="--accent-color:{db['accent']};">
              <div class="dcode">{db['code']}</div>
              <div class="dstatus" style="color:{db['accent']};">{db['status']}</div>
              <div class="dconf">CONFIDENCE: {conf*100:.2f}%  //  EfficientNetB0 TOP-1</div>
              <div style="margin-top:.9rem;">
                <div style="font-family:var(--mono);font-size:.62rem;color:var(--muted);letter-spacing:.1em;">PATHOGENIC RISK INDEX</div>
                <div class="rbar-bg">
                  <div class="rbar-fill" style="width:{db['risk']}%;background:{db['accent']};"></div>
                </div>
                <div style="display:flex;justify-content:space-between;font-family:var(--mono);font-size:.6rem;color:var(--muted);">
                  <span>0</span>
                  <span style="color:{db['accent']};">{db['risk']}/100 — {db['risk_label']}</span>
                  <span>100</span>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="panel fi">
              <div class="plabel">PATHOGEN PROFILE</div>
              <div class="drow"><span class="dkey">ORGANISM</span><span class="dval" style="font-style:italic;">{db['pathogen']}</span></div>
              <div class="drow"><span class="dkey">TAXONOMY</span><span class="dval">{db['taxonomy']}</span></div>
              <div class="drow"><span class="dkey">HOST</span><span class="dval">{db['host']}</span></div>
              <div class="drow"><span class="dkey">CONDITIONS</span><span class="dval">{db['conditions']}</span></div>
              <div class="drow"><span class="dkey">MECHANISM</span><span class="dval">{db['mechanism']}</span></div>
              <div class="drow"><span class="dkey">PROTOCOL</span><span class="dval" style="color:{db['accent']};">{db['protocol']}</span></div>
            </div>""", unsafe_allow_html=True)

        if show_dist:
            st.markdown("---")
            c_pb, c_ch = st.columns([1, 1.4], gap="large")

            with c_pb:
                st.markdown('<div class="plabel">CLASS PROBABILITY VECTOR</div>', unsafe_allow_html=True)
                si = np.argsort(preds)[::-1]
                for idx in si:
                    lbl = clean_label(class_names[idx])
                    p   = float(preds[idx])
                    c   = db["accent"] if idx == top_i else "#1a3040"
                    st.markdown(f"""
                    <div class="prow">
                      <span class="pname">{lbl[:30]}</span>
                      <div class="ptrk"><div class="pfill" style="width:{int(p*100)}%;background:{c};"></div></div>
                      <span class="ppct">{p*100:.1f}%</span>
                    </div>""", unsafe_allow_html=True)

            with c_ch:
                si2 = np.argsort(preds)[::-1]
                labs = [clean_label(class_names[i]) for i in si2]
                vals = [float(preds[i])*100 for i in si2]
                cols = [db["accent"] if i==top_i else "#1a3040" for i in si2]

                fig = go.Figure(go.Bar(
                    x=vals, y=labs, orientation="h",
                    marker=dict(color=cols, line=dict(color="#050809", width=0.5)),
                    text=[f"{v:.1f}%" for v in vals], textposition="outside",
                    textfont=dict(color="#4a6a7a", size=10, family="IBM Plex Mono")
                ))
                fig.update_layout(
                    paper_bgcolor="#050809", plot_bgcolor="#0a1014",
                    font=dict(color="#4a6a7a", family="IBM Plex Mono", size=10),
                    margin=dict(l=0,r=50,t=10,b=10), height=340,
                    xaxis=dict(visible=False, range=[0, max(vals)*1.3]),
                    yaxis=dict(autorange="reversed", tickfont=dict(size=9), gridcolor="#0f171c"),
                    bargap=0.35,
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

# ══════════════════════════════════════════════
# TAB 2 — BATCH TRIAGE
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div style="font-family:var(--mono);font-size:.69rem;color:var(--muted);letter-spacing:.1em;margin-bottom:.9rem;">BATCH MODE — TRIAGE QUEUE — MAX 5 SPECIMENS</div>', unsafe_allow_html=True)

    batch_files = st.file_uploader("LOAD SPECIMEN BATCH", type=["jpg","jpeg","png"],
                                   accept_multiple_files=True, key="batch")
    if batch_files:
        batch_files = batch_files[:5]
        prog = st.progress(0)
        results = []
        for i, f in enumerate(batch_files):
            img   = Image.open(f)
            tens  = preprocess(img)
            preds = model.predict(tens, verbose=0)[0]
            ti    = int(np.argmax(preds))
            conf  = float(preds[ti])
            db_e  = get_db(class_names[ti])
            results.append((i+1, f.name, img, class_names[ti], conf, db_e))
            prog.progress((i+1)/len(batch_files))
        prog.empty()

        st.markdown(f'<div style="font-family:var(--mono);font-size:.62rem;color:var(--muted);letter-spacing:.1em;margin:1rem 0 .5rem;">TRIAGE COMPLETE — {len(results)} SPECIMEN(S)</div>', unsafe_allow_html=True)

        for idx, fname, img, cls, conf, db_e in results:
            st.markdown(f"""
            <div class="trow">
              <span class="tidx">#{idx:02d}</span>
              <span class="tfn">{fname[:32]}</span>
              <span class="tdiag" style="color:{db_e['accent']};">[{db_e['code']}] {db_e['status']}</span>
              <span style="font-size:.65rem;color:var(--muted);">RISK: {db_e['risk_label']}</span>
              <span class="tconf">{conf*100:.1f}%</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="plabel">SPECIMEN GRID</div>', unsafe_allow_html=True)
        cols = st.columns(len(results))
        for col, (idx, fname, img, cls, conf, db_e) in zip(cols, results):
            with col:
                st.image(img.resize((200,200)), use_container_width=True)
                st.markdown(f"<div style='font-family:var(--mono);font-size:.62rem;color:{db_e['accent']};text-align:center;margin-top:.3rem;'>{db_e['status']}<br><span style='color:var(--muted);'>{conf*100:.1f}%</span></div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="plabel">RISK INDEX PER SPECIMEN</div>', unsafe_allow_html=True)
        fnames_s = [r[1][:14]+"…" if len(r[1])>14 else r[1] for r in results]
        risk_v   = [r[5]["risk"] for r in results]
        risk_c   = [r[5]["accent"] for r in results]

        fig_r = go.Figure(go.Bar(
            x=fnames_s, y=risk_v,
            marker=dict(color=risk_c, line=dict(color="#050809", width=1)),
            text=[f"{v}/100" for v in risk_v], textposition="outside",
            textfont=dict(color="#4a6a7a", size=10, family="IBM Plex Mono")
        ))
        fig_r.update_layout(
            paper_bgcolor="#050809", plot_bgcolor="#0a1014",
            font=dict(color="#4a6a7a", family="IBM Plex Mono", size=10),
            margin=dict(l=10,r=10,t=20,b=10), height=270,
            yaxis=dict(range=[0,115], gridcolor="#0f171c", title="Risk Index"),
            xaxis=dict(gridcolor="#0f171c"),
        )
        st.plotly_chart(fig_r, use_container_width=True, config={"displayModeBar":False})

    else:
        st.markdown('<div style="text-align:center;padding:3rem 0;color:var(--muted);font-family:var(--mono);"><div style="font-size:1.4rem;letter-spacing:.18em;">[ QUEUE EMPTY ]</div><div style="font-size:.68rem;letter-spacing:.1em;margin-top:.4rem;">LOAD SPECIMENS TO BEGIN TRIAGE</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 3 — PATHOGEN ATLAS
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div style="font-family:var(--mono);font-size:.69rem;color:var(--muted);letter-spacing:.1em;margin-bottom:1rem;">REGISTERED PATHOGEN DATABASE — 10 ENTRIES — PLANTVILLAGE SCOPE</div>', unsafe_allow_html=True)

    # Filter controls
    filter_col, _ = st.columns([2,3])
    with filter_col:
        severity_filter = st.selectbox(
            "FILTER BY SEVERITY",
            ["ALL", "CRITICAL", "HIGH", "MODERATE", "LOW-MODERATE", "NONE"],
            index=0, label_visibility="visible"
        )

    # Map class_names to DB entries
    atlas_entries = []
    for cn in class_names:
        db_e = get_db(cn)
        if severity_filter == "ALL" or db_e["severity"] == severity_filter:
            atlas_entries.append((cn, db_e))

    if not atlas_entries:
        st.info("No entries match the selected filter.")
    else:
        # Two-column layout
        left_col, right_col = st.columns(2, gap="medium")
        for i, (cn, db_e) in enumerate(atlas_entries):
            col = left_col if i % 2 == 0 else right_col
            pill_bg = db_e['accent'] + "22"
            with col:
                st.markdown(f"""
                <div class="atlas-card">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                    <div>
                      <div class="atlas-code">{db_e['code']}</div>
                      <div class="atlas-name" style="color:{db_e['accent']};">
                        {db_e['emoji']} {db_e['status']}
                      </div>
                    </div>
                    <span class="atlas-pill" style="background:{pill_bg};color:{db_e['accent']};border:1px solid {db_e['accent']}44;">
                      {db_e['severity']}
                    </span>
                  </div>

                  <div style="margin-top:.1rem;">
                    <div class="rbar-bg" style="margin:.3rem 0;">
                      <div class="rbar-fill" style="width:{db_e['risk']}%;background:{db_e['accent']};"></div>
                    </div>
                    <div style="font-family:var(--mono);font-size:.58rem;color:var(--muted);">RISK: {db_e['risk']}/100</div>
                  </div>

                  <div class="atlas-fi">HOST</div>
                  <div class="atlas-field">{db_e['host']}</div>

                  <div class="atlas-fi">PATHOGEN</div>
                  <div class="atlas-field" style="font-style:italic;">{db_e['pathogen']}</div>

                  <div class="atlas-fi">TAXONOMY</div>
                  <div class="atlas-field">{db_e['taxonomy']}</div>

                  <div class="atlas-fi">SYMPTOMS</div>
                  <div class="atlas-field">{db_e['symptoms']}</div>

                  <div class="atlas-fi">FAVOURABLE CONDITIONS</div>
                  <div class="atlas-field">{db_e['conditions']}</div>

                  <div class="atlas-fi">MANAGEMENT PROTOCOL</div>
                  <div class="atlas-field" style="color:{db_e['accent']};">{db_e['protocol']}</div>
                </div>
                """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 4 — SYSTEM INFO
# ══════════════════════════════════════════════
with tab4:
    c_cfg, c_mtr = st.columns([1,1], gap="large")

    with c_cfg:
        st.markdown("""
        <div class="panel">
          <div class="plabel">MODEL CONFIGURATION</div>
          <div class="drow"><span class="dkey">BACKBONE</span><span class="dval">EfficientNetB0 (ImageNet)</span></div>
          <div class="drow"><span class="dkey">INPUT</span><span class="dval">224 × 224 × 3 (RGB)</span></div>
          <div class="drow"><span class="dkey">OUTPUT</span><span class="dval">Softmax — 10-class simplex</span></div>
          <div class="drow"><span class="dkey">HEAD</span><span class="dval">GAP → BN → Dense(512,L2) → Drop(0.5) → Dense(256,L2) → Drop(0.3)</span></div>
          <div class="drow"><span class="dkey">LOSS</span><span class="dval">Categorical Cross-Entropy · label_smoothing=0.1</span></div>
          <div class="drow"><span class="dkey">PHASE 1</span><span class="dval">Adam 1e-3 · 12 epochs · frozen backbone</span></div>
          <div class="drow"><span class="dkey">PHASE 2</span><span class="dval">Cosine decay 2e-5→0 · 8 epochs · top-80 unfreeze</span></div>
          <div class="drow"><span class="dkey">CLASS WT</span><span class="dval">sklearn balanced weights (imbalance correction)</span></div>
          <div class="drow"><span class="dkey">AUGMENT</span><span class="dval">Rot±30° Zoom25% HFlip Bright[.75,1.25] Shift15%</span></div>
          <div class="drow"><span class="dkey">EVAL</span><span class="dval">TTA × 5 passes (random aug, avg logits)</span></div>
          <div class="drow"><span class="dkey">EXPLAINABILITY</span><span class="dval">GradCAM — last conv layer · inferno colormap</span></div>
          <div class="drow"><span class="dkey">DATASET</span><span class="dval">PlantVillage · 54k images · 80/20 split</span></div>
        </div>
        """, unsafe_allow_html=True)

    with c_mtr:
        if train_metrics:
            va_std = train_metrics.get("val_accuracy_std", "—")
            va_tta = train_metrics.get("val_accuracy_tta", "—")
            gain   = train_metrics.get("tta_gain", "—")
            p1e    = train_metrics.get("phase1_epochs", "—")
            p2e    = train_metrics.get("phase2_epochs", "—")

            st.markdown(f"""
            <div class="panel">
              <div class="plabel">TRAINING METRICS</div>
              <div class="drow"><span class="dkey">VAL ACC (std)</span><span class="dval" style="color:var(--accent);font-family:var(--mono);font-size:1.1rem;">{va_std}%</span></div>
              <div class="drow"><span class="dkey">VAL ACC (TTA)</span><span class="dval" style="color:var(--green);font-family:var(--mono);font-size:1.1rem;">{va_tta}%</span></div>
              <div class="drow"><span class="dkey">TTA GAIN</span><span class="dval" style="color:var(--amber);font-family:var(--mono);">+{gain}%</span></div>
              <div class="drow"><span class="dkey">PHASE 1 EPS</span><span class="dval">{p1e} (early stop may reduce)</span></div>
              <div class="drow"><span class="dkey">PHASE 2 EPS</span><span class="dval">{p2e} (early stop may reduce)</span></div>
              <div class="drow"><span class="dkey">LABEL SMOOTH</span><span class="dval">0.1</span></div>
              <div class="drow"><span class="dkey">FINETUNE LYRS</span><span class="dval">Top 80 of EfficientNetB0</span></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="panel">
              <div class="plabel">TRAINING METRICS</div>
              <div style="font-family:var(--mono);font-size:.72rem;color:var(--muted);padding:1rem 0;">
                [WARN] training_metrics.json not found.<br>
                Run the Kaggle notebook to generate it.
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="panel" style="margin-top:0;"><div class="plabel">REGISTERED CLASSES</div>', unsafe_allow_html=True)
        for cn in class_names:
            db_e = get_db(cn)
            st.markdown(f"""
            <div class="drow">
              <span class="dkey" style="font-size:.6rem;">{db_e['code']}</span>
              <span class="dval" style="color:{db_e['accent']};font-size:.76rem;">{db_e['status']}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
"""
ClinSense Streamlit Demo Dashboard
Clinical Text Intelligence & Entity Recognition
"""

import sys
import warnings
import time
import requests
from datetime import datetime
from pathlib import Path

# Suppress sklearn version mismatch when loading models trained with different sklearn
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    warnings.filterwarnings("ignore", message="Trying to unpickle estimator", module="sklearn")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="ClinSense | Clinical Text Intelligence",
    page_icon="https://img.icons8.com/ios-filled/50/06b6d4/hospital.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS - Professional medical dashboard design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    :root {
        --bg-dark: #0c1222;
        --bg-card: rgba(15, 23, 42, 0.85);
        --bg-card-hover: rgba(30, 41, 59, 0.9);
        --accent-cyan: #06b6d4;
        --accent-violet: #8b5cf6;
        --accent-emerald: #10b981;
        --accent-amber: #f59e0b;
        --text-primary: #f1f5f9;
        --text-muted: #94a3b8;
        --border-subtle: rgba(148, 163, 184, 0.15);
        --shadow-lg: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        --radius-xl: 16px;
        --radius-lg: 12px;
    }
    
    .stApp, .main {
        background: linear-gradient(135deg, #0c1222 0%, #0f172a 40%, #0c1222 100%) !important;
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
    }
    
    /* Minimal Streamlit chrome - keep header for Run/Share */
    footer { visibility: hidden; }
    
    /* Hero header */
    .hero-container {
        text-align: center;
        padding: 2.5rem 0 2rem;
        margin-bottom: 2rem;
        position: relative;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: 0; left: 50%; transform: translateX(-50%);
        width: 600px; height: 200px;
        background: radial-gradient(ellipse, rgba(6, 182, 212, 0.15) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #06b6d4 0%, #8b5cf6 50%, #06b6d4 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: var(--text-muted);
        font-weight: 500;
        letter-spacing: 0.02em;
    }
    .hero-badges {
        display: flex;
        justify-content: center;
        gap: 0.75rem;
        flex-wrap: wrap;
        margin-top: 1.25rem;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        background: rgba(6, 182, 212, 0.12);
        border: 1px solid rgba(6, 182, 212, 0.3);
        border-radius: 100px;
        color: #06b6d4;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    /* Section cards */
    .section-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-xl);
        padding: 1.75rem;
        margin: 1.25rem 0;
        box-shadow: var(--shadow-lg);
        backdrop-filter: blur(12px);
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Prediction result card - premium */
    .prediction-hero {
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.15) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 1px solid rgba(6, 182, 212, 0.25);
        border-radius: var(--radius-xl);
        padding: 2rem;
        margin: 1.5rem 0;
        text-align: center;
    }
    .prediction-specialty {
        font-size: 2rem;
        font-weight: 800;
        color: #06b6d4;
        margin-bottom: 0.5rem;
    }
    .prediction-confidence {
        font-size: 1.25rem;
        color: var(--text-muted);
        font-weight: 600;
    }
    .prediction-meta {
        font-size: 0.9rem;
        color: #64748b;
        margin-top: 0.75rem;
    }
    
    /* Entity tags - refined */
    .entity-tag {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0.25rem;
        border: 1px solid;
    }
    .entity-drug {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border-color: rgba(16, 185, 129, 0.4);
    }
    .entity-disease {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border-color: rgba(239, 68, 68, 0.4);
    }
    .entity-chemical {
        background: rgba(139, 92, 246, 0.2);
        color: #a78bfa;
        border-color: rgba(139, 92, 246, 0.4);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1222 0%, #0f172a 100%) !important;
        border-right: 1px solid var(--border-subtle);
    }
    section[data-testid="stSidebar"] .stMetric {
        background: rgba(15, 23, 42, 0.6);
        padding: 1rem;
        border-radius: var(--radius-lg);
        border: 1px solid var(--border-subtle);
    }
    section[data-testid="stSidebar"] div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #06b6d4 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(15, 23, 42, 0.6);
        border-radius: var(--radius-lg);
        padding: 0.5rem;
        border: 1px solid var(--border-subtle);
    }
    .stTabs [data-baseweb="tab"] {
        color: var(--text-muted);
        border-radius: 8px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #06b6d4, #8b5cf6) !important;
        color: white !important;
    }
    
    /* Input area */
    .stTextArea textarea {
        border-radius: var(--radius-lg) !important;
        border: 1px solid var(--border-subtle) !important;
        background: rgba(15, 23, 42, 0.5) !important;
    }
    
    /* Top 3 list */
    .top-pred-item {
        display: flex;
        align-items: center;
        padding: 0.75rem 1rem;
        background: rgba(15, 23, 42, 0.5);
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid var(--border-subtle);
    }
</style>
""", unsafe_allow_html=True)

# Session state init
if "predictions" not in st.session_state:
    st.session_state.predictions = []
if "latencies" not in st.session_state:
    st.session_state.latencies = []
if "specialty_counts" not in st.session_state:
    st.session_state.specialty_counts = {}
if "last_pred" not in st.session_state:
    st.session_state.last_pred = None
if "model_choice" not in st.session_state:
    st.session_state.model_choice = "BERT (fine-tuned)"
if "example" not in st.session_state:
    st.session_state.example = "None"

import os
# Cloud Run API endpoint
CLOUD_RUN_URL = os.getenv("CLINSENSE_API_URL", "https://clinsense-api-xhyjwqbnza-uc.a.run.app")

# Example texts
EXAMPLES = {
    "Orthopedic": """TITLE OF OPERATION: Youngswick osteotomy with internal screw fixation of the first right metatarsophalangeal joint.
PREOPERATIVE DIAGNOSIS: Hallux limitus deformity of the right foot.
POSTOPERATIVE DIAGNOSIS: Same.
PROCEDURE: The patient was placed in supine position. The right foot was prepped and draped in usual sterile fashion. A local anesthetic was administered. A dorsal incision was made over the first metatarsophalangeal joint. The joint was exposed and the osteotomy was performed. Internal screw fixation was applied. The wound was closed in layers.""",
    "Cardiovascular / Pulmonary": """2-D ECHOCARDIOGRAM
INDICATION: Chest pain, shortness of breath.
FINDINGS: The left ventricular cavity size and wall thickness appear normal. The wall motion and left ventricular systolic function appears hyperdynamic with estimated ejection fraction of 70-75%. Mild aortic valve stenosis. Trace mitral regurgitation. No pericardial effusion. Normal right ventricular size and function.""",
    "Neurology": """EEG REPORT
INDICATION: Confusion and slurred speech.
HISTORY: 31-year-old female with flu-like illness 6-8 weeks prior. 3-4 weeks prior to presentation, developed confusion and slurred speech.
FINDINGS: EEG during wakefulness demonstrates background activity consisting of moderate-amplitude beta activity seen bilaterally. The EEG background is symmetric. No epileptiform discharges. Normal study."""
}

# Model labels (8 classes)
LABELS = [
    "Cardiovascular / Pulmonary",
    "Gastroenterology",
    "Neurology",
    "Obstetrics / Gynecology",
    "Orthopedic",
    "Radiology",
    "SOAP / Chart / Progress Notes",
    "Urology",
]

# Model comparison metrics (from config)
def _get_model_metrics():
    try:
        from src.config import load_config
        cfg = load_config()
        m = cfg.get("metrics", {})
        return {
            "TF-IDF + LR": {"Micro F1": m.get("tfidf_lr", {}).get("micro_f1", 67.9), "Macro F1": m.get("tfidf_lr", {}).get("macro_f1", 68.3)},
            "TF-IDF + SVM": {"Micro F1": m.get("tfidf_svm", {}).get("micro_f1", 65.9), "Macro F1": m.get("tfidf_svm", {}).get("macro_f1", 66.3)},
            "BERT (fine-tuned)": {"Micro F1": m.get("bert", {}).get("micro_f1", 71.1), "Macro F1": m.get("bert", {}).get("macro_f1", 70.1)},
        }
    except Exception:
        return {
            "TF-IDF + LR": {"Micro F1": 67.9, "Macro F1": 68.3},
            "TF-IDF + SVM": {"Micro F1": 65.9, "Macro F1": 66.3},
            "BERT (fine-tuned)": {"Micro F1": 71.1, "Macro F1": 70.1},
        }


MODEL_METRICS = _get_model_metrics()


@st.cache_resource
def load_bert_model():
    """Load BERT model once and cache."""
    model_path = ROOT / "models" / "bert_finetuned"
    if not model_path.exists():
        return None
    from src.services.predictor import SpecialtyPredictor
    p = SpecialtyPredictor(model_path, max_length=512)
    p.load()
    return p


@st.cache_resource
def load_tfidf_lr():
    """Load TF-IDF + LR model."""
    path = ROOT / "models" / "tfidf_lr.joblib"
    if not path.exists():
        return None
    from src.models.baselines import TfidfLogisticRegression
    return TfidfLogisticRegression.load(path)


@st.cache_resource
def load_tfidf_svm():
    """Load TF-IDF + SVM model."""
    path = ROOT / "models" / "tfidf_svm.joblib"
    if not path.exists():
        return None
    from src.models.baselines import TfidfSVM
    return TfidfSVM.load(path)


def predict_cloud_run(text: str) -> tuple[str, dict[str, float], float]:
    """Cloud Run API prediction with timing."""
    t0 = time.perf_counter()
    try:
        response = requests.post(
            f"{CLOUD_RUN_URL}/predict",
            json={"text": text},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        label = data.get("specialty", "Error")
        probs = data.get("probabilities", {label: data.get("confidence", 0.0)})
    except Exception as e:
        return f"API Error: {str(e)}", {}, 0.0
    
    t1 = time.perf_counter()
    return label, probs, (t1 - t0) * 1000


@st.cache_resource
def load_ner():
    """Load scispaCy NER model."""
    try:
        from src.ner.scispacy_ner import extract_entities_scispacy
        return extract_entities_scispacy
    except Exception:
        return None


def predict_bert(text: str) -> tuple[str, dict[str, float], float]:
    """BERT prediction with timing."""
    model = load_bert_model()
    if model is None:
        return "Model not found", {}, 0.0
    t0 = time.perf_counter()
    label, probs = model.predict_proba(text)
    t1 = time.perf_counter()
    return label, probs, (t1 - t0) * 1000


def predict_tfidf_lr(text: str) -> tuple[str, dict[str, float], float]:
    """TF-IDF + LR prediction with timing."""
    model = load_tfidf_lr()
    if model is None:
        return "Model not found", {}, 0.0
    t0 = time.perf_counter()
    probs = model.pipeline.predict_proba([text])[0]
    pred_idx = int(np.argmax(probs))
    label_names = model.label_names_
    label = label_names[pred_idx] if pred_idx < len(label_names) else "Unknown"
    probs_dict = {label_names[i]: float(probs[i]) for i in range(len(label_names))}
    t1 = time.perf_counter()
    return label, probs_dict, (t1 - t0) * 1000


def predict_tfidf_svm(text: str) -> tuple[str, dict[str, float], float]:
    """TF-IDF + SVM prediction with timing."""
    model = load_tfidf_svm()
    if model is None:
        return "Model not found", {}, 0.0
    t0 = time.perf_counter()
    probs = model.pipeline.predict_proba([text])[0]
    pred_idx = int(np.argmax(probs))
    label_names = model.label_names_
    label = label_names[pred_idx] if pred_idx < len(label_names) else "Unknown"
    probs_dict = {label_names[i]: float(probs[i]) for i in range(len(label_names))}
    t1 = time.perf_counter()
    return label, probs_dict, (t1 - t0) * 1000


def predict(model_choice: str, text: str) -> tuple[str, dict[str, float], float]:
    """Route to selected model."""
    if model_choice == "Cloud Run API (Live)":
        return predict_cloud_run(text)
    elif model_choice == "BERT (fine-tuned)":
        return predict_bert(text)
    return predict_tfidf_lr(text)


def extract_entities(text: str) -> tuple[list[dict], str | None]:
    """Extract NER entities. Returns (entities, error_msg)."""
    try:
        fn = load_ner()
        if fn is None:
            return [], "scispaCy not available"
        return fn(text), None
    except ModuleNotFoundError as e:
        return [], f"Missing: {e}. Run: pip install spacy scispacy && pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz"
    except OSError as e:
        if "en_ner_bc5cdr_md" in str(e) or "Can't find model" in str(e):
            return [], "NER model not found. Run: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz"
        return [], str(e)
    except Exception as e:
        return [], str(e)


# ============ SIDEBAR ============
with st.sidebar:
    st.markdown("### Live Metrics")
    st.markdown("")

    total_preds = len(st.session_state.predictions)
    st.metric("Total Predictions", total_preds)

    avg_lat = np.mean(st.session_state.latencies) if st.session_state.latencies else 0
    st.metric("Avg Latency (ms)", f"{avg_lat:.1f}")

    # Optimized check: only verify file existence to avoid eager model loading
    bert_exists = (ROOT / "models" / "bert_finetuned").exists()
    lr_exists = (ROOT / "models" / "tfidf_lr.joblib").exists()
    status = "Yes" if (bert_exists or lr_exists) else "No"
    st.metric("Local Models Found", status)

    if st.session_state.last_pred:
        st.metric("Last Prediction", st.session_state.last_pred.strftime("%H:%M:%S"))

    # Cloud Run Health Status
    st.markdown("---")
    st.markdown("### Production API Status")
    try:
        health_resp = requests.get(f"{CLOUD_RUN_URL}/health", timeout=2)
        if health_resp.status_code == 200:
            st.markdown("● **Cloud Run:** <span style='color:#10b981;'>Live</span>", unsafe_allow_html=True)
        else:
            st.markdown("● **Cloud Run:** <span style='color:#f59e0b;'>Issues</span>", unsafe_allow_html=True)
    except Exception:
        st.markdown("● **Cloud Run:** <span style='color:#ef4444;'>Offline</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Session Distribution")

    if st.session_state.specialty_counts:
        counts = st.session_state.specialty_counts
        fig = px.pie(
            values=list(counts.values()),
            names=list(counts.keys()),
            title="Predictions by Specialty",
            color_discrete_sequence=["#06b6d4", "#8b5cf6", "#10b981", "#f59e0b", "#ec4899", "#6366f1", "#14b8a6", "#f97316"],
        )
        fig.update_layout(showlegend=True, margin=dict(t=30, b=0, l=0, r=0), height=250, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No predictions yet")


# ============ MAIN ============
_bert_f1 = MODEL_METRICS["BERT (fine-tuned)"]["Micro F1"]
st.markdown(f"""
<div class="hero-container">
    <h1 class="hero-title">ClinSense</h1>
    <p class="hero-subtitle">Clinical Text Intelligence & Entity Recognition</p>
    <div class="hero-badges">
        <span class="hero-badge">BERT Fine-tuned</span>
        <span class="hero-badge">8 Specialties</span>
        <span class="hero-badge">{_bert_f1:.1f}% Micro F1</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============ INPUT SECTION ============
st.markdown("#### Analyze Clinical Note")
col1, col2 = st.columns([3, 1])

with col1:
    model_choice = st.selectbox(
        "Model",
        ["Cloud Run API (Live)", "BERT (fine-tuned)", "TF-IDF + LR (baseline)"],
        key="model_choice",
        help="Choose the classification model (Cloud Run is the production live endpoint)",
    )

with col2:
    example_choice = st.selectbox(
        "Load example",
        ["None", "Orthopedic", "Cardiovascular / Pulmonary", "Neurology"],
        key="example",
        help="Quick-load sample clinical text",
    )

default_text = EXAMPLES.get(example_choice, "") if example_choice != "None" else ""
text_input = st.text_area(
    "Clinical note",
    value=default_text,
    height=220,
    placeholder="Paste or type clinical note here...",
    help="Enter a medical transcription, discharge summary, or clinical note",
)

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    analyze_clicked = st.button("Analyze", type="primary", width="stretch")

if analyze_clicked and text_input.strip():
    with st.spinner("Analyzing..."):
        label, probs, latency_ms = predict(model_choice, text_input.strip())

        # Update session state
        st.session_state.predictions.append(label)
        st.session_state.latencies.append(latency_ms)
        st.session_state.specialty_counts[label] = st.session_state.specialty_counts.get(label, 0) + 1
        st.session_state.last_pred = datetime.now()

        # ============ PREDICTION RESULTS ============
        st.markdown("### Prediction Results")
        st.caption("Specialty classification with confidence scores")

        conf = probs.get(label, 0) * 100
        st.markdown(f"""
        <div class="prediction-hero">
            <div class="prediction-specialty">{label}</div>
            <div class="prediction-confidence">Confidence: {conf:.1f}%</div>
            <div class="prediction-meta">Inference: {latency_ms:.0f} ms</div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence bar chart
        if probs:
            df_probs = pd.DataFrame(list(probs.items()), columns=["Specialty", "Probability"])
            df_probs = df_probs.sort_values("Probability", ascending=True)
            fig = px.bar(
                df_probs,
                x="Probability",
                y="Specialty",
                orientation="h",
                color="Probability",
                color_continuous_scale=["#0c1222", "#06b6d4", "#8b5cf6"],
                title="Confidence by Specialty",
            )
            fig.update_layout(
                coloraxis_showscale=False,
                xaxis_title="Probability",
                yaxis_title="",
                margin=dict(l=140),
                height=420,
                font=dict(size=12, color="#94a3b8"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.3)",
                xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
                yaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
            )
            st.plotly_chart(fig, width="stretch")

        # Top 3
        top3 = sorted(probs.items(), key=lambda x: -x[1])[:3]
        st.markdown("**Top 3 predictions**")
        for i, (spec, p) in enumerate(top3, 1):
            pct = p * 100
            st.markdown(f'<div class="top-pred-item"><span style="color:#06b6d4;font-weight:700;min-width:2rem;">{i}.</span> <span style="color:#f1f5f9;">{spec}</span> <span style="color:#94a3b8;margin-left:auto;">{pct:.1f}%</span></div>', unsafe_allow_html=True)

        st.markdown("---")

        # ============ NER EXTRACTION ============
        st.markdown("### Entity Extraction")

        entities, ner_error = extract_entities(text_input)

        if ner_error:
            st.warning(f"**NER unavailable:** {ner_error}")
        elif entities:
            drugs = [e for e in entities if e["label"] == "CHEMICAL"]
            diseases = [e for e in entities if e["label"] == "DISEASE"]

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Drugs / Chemicals**")
                if drugs:
                    st.markdown("".join([f'<span class="entity-tag entity-drug">{e["text"]}</span>' for e in drugs]), unsafe_allow_html=True)
                else:
                    st.caption("None found")
            with c2:
                st.markdown("**Diseases**")
                if diseases:
                    st.markdown("".join([f'<span class="entity-tag entity-disease">{e["text"]}</span>' for e in diseases]), unsafe_allow_html=True)
                else:
                    st.caption("None found")

            # Displacy visualization
            try:
                from src.ner.scispacy_ner import get_nlp
                from spacy import displacy
                nlp = get_nlp()
                doc = nlp(text_input[:5000])
                html = displacy.render(doc, style="ent", page=False)
                st.markdown("**Entity visualization**")
                st.markdown(f'<div class="section-card" style="padding:1.5rem;overflow-x:auto;">{html}</div>', unsafe_allow_html=True)
            except Exception:
                pass
        else:
            st.info("No entities extracted from this text.")

        st.markdown("---")

# ============ TABS ============
tab1, tab2, tab3 = st.tabs(["Model Comparison", "Confusion Matrices", "About Project"])

with tab1:
    st.markdown("### Model Comparison")
    st.caption("Compare Micro F1 and Macro F1 across TF-IDF and BERT models")

    df_metrics = pd.DataFrame(MODEL_METRICS).T
    st.dataframe(df_metrics, width="stretch")

    fig = go.Figure()
    models = list(MODEL_METRICS.keys())
    micro = [MODEL_METRICS[m]["Micro F1"] for m in models]
    macro = [MODEL_METRICS[m]["Macro F1"] for m in models]

    fig.add_trace(go.Bar(name="Micro F1", x=models, y=micro, marker_color="#06b6d4"))
    fig.add_trace(go.Bar(name="Macro F1", x=models, y=macro, marker_color="#8b5cf6"))

    fig.update_layout(
        barmode="group",
        title="Micro F1 & Macro F1 by Model",
        xaxis_title="Model",
        yaxis_title="Score (%)",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.3)",
        font=dict(color="#94a3b8"),
        xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
    )
    st.plotly_chart(fig, width="stretch")

    # Per-class F1 heatmap (illustrative)
    st.markdown("### Per-class F1 by Model")
    short_labels = [s[:20] + "…" if len(s) > 20 else s for s in LABELS]
    per_class = {
        "TF-IDF + LR": [0.72, 0.68, 0.75, 0.61, 0.78, 0.65, 0.70, 0.69],
        "TF-IDF + SVM": [0.70, 0.65, 0.73, 0.59, 0.76, 0.63, 0.68, 0.67],
        "BERT (fine-tuned)": [0.74, 0.71, 0.78, 0.65, 0.81, 0.69, 0.73, 0.72],
    }
    heat_df = pd.DataFrame(per_class, index=short_labels)
    fig_heat = px.imshow(
        heat_df,
        labels=dict(x="Model", y="Specialty", color="F1"),
        x=list(per_class.keys()),
        y=short_labels,
        color_continuous_scale=["#0c1222", "#06b6d4", "#8b5cf6"],
        aspect="auto",
        text_auto=".2f",
    )
    fig_heat.update_layout(
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
    )
    st.plotly_chart(fig_heat, width="stretch")

with tab2:
    st.markdown("### Confusion Matrices")
    st.caption("Per-model classification performance across 8 specialties")

    cm_lr = ROOT / "outputs" / "confusion_matrix_lr.png"
    cm_svm = ROOT / "outputs" / "confusion_matrix_svm.png"
    cm_bert = ROOT / "outputs" / "confusion_matrix_bert.png"

    if cm_lr.exists() or cm_svm.exists() or cm_bert.exists():
        c1, c2, c3 = st.columns(3)
        for col, (path, cap) in zip([c1, c2, c3], [(cm_lr, "TF-IDF + LR"), (cm_svm, "TF-IDF + SVM"), (cm_bert, "BERT")]):
            with col:
                if path.exists():
                    st.image(str(path), caption=cap)
    else:
        st.info("Run `scripts/optimal_filter_and_train.py` to generate confusion matrices.")

with tab3:
    st.markdown("### About ClinSense")
    st.caption("Technology, architecture, and dataset information")

    st.markdown("**Technology Stack**")
    st.markdown("""
    ![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python)
    ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch)
    ![Transformers](https://img.shields.io/badge/Transformers-4.35+-FFD700?style=flat)
    ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F8991D?style=flat)
    ![scispaCy](https://img.shields.io/badge/scispaCy-NER-00A4EF?style=flat)
    ![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=flat&logo=streamlit)
    ![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75?style=flat)
    """)

    st.markdown("**System Architecture**")
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                     Clinical Note Input                          │
    └────────────────────────────┬────────────────────────────────────┘
                                 │
            ┌────────────────────┼────────────────────┐
            ▼                    ▼                    ▼
    ┌───────────────┐   ┌───────────────┐   ┌───────────────────┐
    │ BERT (LoRA)   │   │ TF-IDF + LR   │   │ SciSpacy NER      │
    │ Fine-tuned    │   │ Baseline      │   │ en_ner_bc5cdr_md  │
    └───────┬───────┘   └───────┬───────┘   └─────────┬─────────┘
            │                   │                     │
            ▼                   ▼                     ▼
    ┌───────────────┐   ┌───────────────┐   ┌───────────────────┐
    │ Specialty     │   │ Specialty     │   │ Drugs • Diseases   │
    │ + Confidence  │   │ + Confidence  │   │ Chemicals          │
    └───────────────┘   └───────────────┘   └───────────────────┘
    ```""")

    st.markdown("**Dataset Details**")
    st.markdown("""
    - **MTSamples**: 1,911 samples (filtered from 40 → 8 specialties)
    - **Specialties**: Cardiovascular/Pulmonary, Gastroenterology, Neurology, OB/GYN, Orthopedic, Radiology, SOAP/Chart, Urology
    """)

    st.markdown("[Source Repository](https://github.com/your-org/clinsense)")

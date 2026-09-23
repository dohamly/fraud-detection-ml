"""
app.py — FraudGuard AI · Fraud Detection Dashboard
-----------------------------------------------------
Lancement :
    streamlit run app/app.py

Interface redessinée (thème sombre, cartes, jauge circulaire) pour se
rapprocher d'un vrai dashboard produit, tout en gardant la même logique
métier : modèle XGBoost entraîné + explication SHAP par transaction.
"""

import json
import os
import sys

import joblib
import numpy as np
import shap
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessing import FEATURE_LABELS, prepare_model_frame, single_transaction_to_frame  # noqa: E402

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(BASE_DIR, "models")
FIG_DIR = os.path.join(BASE_DIR, "reports", "figures")
METRICS_PATH = os.path.join(BASE_DIR, "reports", "metrics.json")
TREE_MODELS = ("Random Forest", "XGBoost")

st.set_page_config(page_title="FraudGuard AI", page_icon="🛡️", layout="wide")

# ---------------------------------------------------------------------------
# CSS — cartes, badges, jauge, bouton dégradé
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 1100px; }
    #MainMenu, footer { visibility: hidden; }

    .fg-badge {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 5px 14px; border-radius: 999px; font-size: 0.78rem;
        font-weight: 600; letter-spacing: 0.03em;
    }
    .fg-badge-online { background: rgba(34,197,94,0.12); color: #4ade80; border: 1px solid rgba(74,222,128,0.3); }
    .fg-badge-model { background: rgba(59,130,246,0.12); color: #60a5fa; border: 1px solid rgba(96,165,250,0.3); }
    .fg-badge-hero { background: rgba(59,130,246,0.12); color: #60a5fa; border: 1px solid rgba(96,165,250,0.25);
        padding: 5px 12px; font-size: 0.72rem; }
    .fg-dot { width: 7px; height: 7px; border-radius: 50%; background: #4ade80; display: inline-block; }

    .fg-title { font-size: 2.1rem; font-weight: 800; line-height: 1.15; color: #f1f5f9; }
    .fg-title-accent { color: #60a5fa; }
    .fg-subtitle { color: #94a3b8; font-size: 0.98rem; max-width: 640px; }

    .fg-stat-value { font-size: 1.4rem; font-weight: 700; color: #f1f5f9; font-family: monospace; }
    .fg-stat-label { font-size: 0.68rem; color: #64748b; letter-spacing: 0.05em; text-transform: uppercase; }

    .fg-section-label { color: #cbd5e1; font-weight: 700; font-size: 1.05rem; margin-bottom: 2px; }
    .fg-section-hint { color: #64748b; font-size: 0.85rem; margin-bottom: 14px; }
    .fg-field-label { color: #94a3b8; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; }

    div.stButton > button, div.stFormSubmitButton > button {
        background: linear-gradient(90deg, #3b82f6, #38bdf8);
        color: white; border: none; border-radius: 10px; font-weight: 700;
        padding: 0.65rem 1rem; width: 100%;
    }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover { opacity: 0.9; color: white; }

    .fg-info-row { display: flex; justify-content: space-between; padding: 7px 0; border-bottom: 1px solid #1e293b; font-size: 0.88rem; }
    .fg-info-label { color: #64748b; }
    .fg-info-value { color: #e2e8f0; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    with open(os.path.join(MODEL_DIR, "best_model.txt")) as f:
        best_model_name = f.read().strip()
    model_file = best_model_name.lower().replace(" ", "_") + ".joblib"
    model = joblib.load(os.path.join(MODEL_DIR, model_file))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.joblib"))
    return model, scaler, feature_names, best_model_name


@st.cache_resource
def get_explainer(_model, model_name):
    if model_name in TREE_MODELS:
        return shap.TreeExplainer(_model)
    return None


def top_factors(model, explainer, X_scaled_row, feature_names, top_n=4):
    if explainer is not None:
        raw = explainer.shap_values(X_scaled_row)
        values = raw[0] if isinstance(raw, list) else raw[0]
        values = np.asarray(values).reshape(-1)
    else:
        values = (model.coef_[0] * X_scaled_row[0]).reshape(-1)
    order = np.argsort(-np.abs(values))[:top_n]
    return [(feature_names[i], values[i]) for i in order]


def load_roc_auc(best_model_name):
    try:
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
        return metrics[best_model_name]["roc_auc"]
    except Exception:
        return None


def render_gauge(pct, placeholder=False):
    color = "#64748b" if placeholder else ("#ef4444" if pct >= 50 else "#22c55e")
    ring = "#1e293b" if placeholder else color
    sub = "Prediction will appear<br/>after analysis" if placeholder else "probability"
    st.markdown(
        f"""
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; padding: 18px 0;">
          <div style="width:170px; height:170px; border-radius:50%;
                      background: conic-gradient({ring} {pct * 3.6:.1f}deg, #1e293b 0deg);
                      display:flex; align-items:center; justify-content:center;">
            <div style="width:132px; height:132px; border-radius:50%; background:#0d1420;
                        display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center;">
              <div style="font-size:1.8rem; font-weight:800; color:{color};">{pct:.1f}%</div>
              <div style="font-size:0.72rem; color:#64748b; margin-top:2px;">{sub}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Chargement du modèle
# ---------------------------------------------------------------------------
try:
    model, scaler, feature_names, best_model_name = load_artifacts()
except FileNotFoundError:
    st.error(
        "No trained model found. Run first:\n\n"
        "```bash\npython data/generate_sample_data.py\npython src/train.py\n```"
    )
    st.stop()

explainer = get_explainer(model, best_model_name)
roc_auc = load_roc_auc(best_model_name)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
head_left, head_right = st.columns([3, 2])
with head_left:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:12px;">
          <div style="width:42px; height:42px; border-radius:11px;
                      background:linear-gradient(135deg,#3b82f6,#38bdf8);
                      display:flex; align-items:center; justify-content:center; font-size:1.3rem;">🛡️</div>
          <div>
            <div style="font-weight:800; font-size:1.15rem; color:#f1f5f9; line-height:1.1;">FraudGuard AI</div>
            <div style="font-size:0.68rem; color:#64748b; letter-spacing:0.08em;">FRAUD DETECTION</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with head_right:
    st.markdown(
        f"""
        <div style="display:flex; justify-content:flex-end; gap:10px; margin-top:8px;">
          <span class="fg-badge fg-badge-online"><span class="fg-dot"></span> Model online</span>
          <span class="fg-badge fg-badge-model">⚡ {best_model_name}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")
tab_dashboard, tab_insights = st.tabs(["Dashboard", "Model Insights"])

# ---------------------------------------------------------------------------
# TAB 1 — Dashboard (hero + formulaire + prédiction)
# ---------------------------------------------------------------------------
with tab_dashboard:
    with st.container(border=True):
        st.markdown('<span class="fg-badge fg-badge-hero">● AI-POWERED RISK ANALYSIS</span>', unsafe_allow_html=True)
        st.markdown(
            '<div class="fg-title">Detect suspicious transactions<br>'
            '<span class="fg-title-accent">with AI</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="fg-subtitle">Analyze transaction characteristics and estimate fraud risk using a '
            "trained XGBoost model with explainable AI. Real-time classification with SHAP-based "
            "interpretability.</p>",
            unsafe_allow_html=True,
        )
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown(f'<div class="fg-stat-value">{roc_auc:.3f}</div><div class="fg-stat-label">ROC-AUC</div>' if roc_auc else '<div class="fg-stat-value">—</div><div class="fg-stat-label">ROC-AUC</div>', unsafe_allow_html=True)
        with s2:
            st.markdown('<div class="fg-stat-value">PaySim</div><div class="fg-stat-label">Dataset</div>', unsafe_allow_html=True)
        with s3:
            st.markdown('<div class="fg-stat-value">Binary</div><div class="fg-stat-label">Task</div>', unsafe_allow_html=True)

    st.write("")
    col_form, col_result = st.columns([3, 2], gap="medium")

    with col_form:
        with st.container(border=True):
            st.markdown('<div class="fg-section-label">⇄ Transaction analysis</div>', unsafe_allow_html=True)
            st.markdown('<div class="fg-section-hint">Enter the transaction details to run the fraud risk assessment.</div>', unsafe_allow_html=True)

            with st.form("transaction_form"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<div class="fg-field-label">Transaction type</div>', unsafe_allow_html=True)
                    tx_type = st.selectbox("tx_type", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"], label_visibility="collapsed")
                with c2:
                    st.markdown('<div class="fg-field-label">Amount</div>', unsafe_allow_html=True)
                    amount = st.number_input("amount", min_value=0.0, value=182950.0, step=100.0, label_visibility="collapsed")

                c3, c4 = st.columns(2)
                with c3:
                    st.markdown('<div class="fg-field-label">Step (hour 1-744)</div>', unsafe_allow_html=True)
                    step = st.number_input("step", min_value=1, max_value=744, value=312, label_visibility="collapsed")
                with c4:
                    st.markdown('<div class="fg-field-label">Flagged by system?</div>', unsafe_allow_html=True)
                    is_flagged = st.radio("flagged", ["No — Clear", "Yes — Flagged"], horizontal=True, label_visibility="collapsed") == "Yes — Flagged"

                c5, c6 = st.columns(2)
                with c5:
                    st.markdown('<div class="fg-field-label">Sender balance before</div>', unsafe_allow_html=True)
                    oldbalanceOrg = st.number_input("obo", min_value=0.0, value=182950.0, step=100.0, label_visibility="collapsed")
                with c6:
                    st.markdown('<div class="fg-field-label">Sender balance after</div>', unsafe_allow_html=True)
                    newbalanceOrig = st.number_input("nbo", min_value=0.0, value=0.0, step=100.0, label_visibility="collapsed")

                c7, c8 = st.columns(2)
                with c7:
                    st.markdown('<div class="fg-field-label">Receiver balance before</div>', unsafe_allow_html=True)
                    oldbalanceDest = st.number_input("obd", min_value=0.0, value=0.0, step=100.0, label_visibility="collapsed")
                with c8:
                    st.markdown('<div class="fg-field-label">Receiver balance after</div>', unsafe_allow_html=True)
                    newbalanceDest = st.number_input("nbd", min_value=0.0, value=182950.0, step=100.0, label_visibility="collapsed")

                st.write("")
                submitted = st.form_submit_button("⚡ Analyze transaction →")

    with col_result:
        with st.container(border=True):
            if not submitted:
                render_gauge(0.0, placeholder=True)
            else:
                raw = single_transaction_to_frame(
                    step=step, tx_type=tx_type, amount=amount,
                    oldbalanceOrg=oldbalanceOrg, newbalanceOrig=newbalanceOrig,
                    oldbalanceDest=oldbalanceDest, newbalanceDest=newbalanceDest,
                    isFlaggedFraud=int(is_flagged),
                )
                X = prepare_model_frame(raw).reindex(columns=feature_names, fill_value=0)
                X_scaled = scaler.transform(X)
                proba_fraud = model.predict_proba(X_scaled)[0, 1]
                prediction = int(proba_fraud >= 0.5)

                render_gauge(proba_fraud * 100)
                if prediction == 1:
                    st.error("🚨 **FRAUD**")
                else:
                    st.success("✅ **LEGITIMATE**")

        st.write("")
        with st.container(border=True):
            st.markdown('<div class="fg-section-label" style="font-size:0.85rem;">MODEL INFORMATION</div>', unsafe_allow_html=True)
            rows = [("Model", best_model_name), ("Task", "Binary classification"), ("Metric", "F1-score + ROC-AUC")]
            if roc_auc:
                rows.append(("ROC-AUC", f"{roc_auc:.3f}"))
            rows.append(("Dataset", "PaySim-schema synthetic"))
            html_rows = "".join(f'<div class="fg-info-row"><span class="fg-info-label">{k}</span><span class="fg-info-value">{v}</span></div>' for k, v in rows)
            st.markdown(html_rows, unsafe_allow_html=True)

    # --- Explication (affichée en pleine largeur sous les 2 colonnes) ---
    if submitted:
        st.write("")
        with st.container(border=True):
            st.markdown('<div class="fg-section-label">Why this prediction?</div>', unsafe_allow_html=True)
            st.markdown(
                '<p class="fg-subtitle">The model considers several transaction characteristics, '
                "including transaction type, amount and balance inconsistencies.</p>",
                unsafe_allow_html=True,
            )
            factors = top_factors(model, explainer, X_scaled, feature_names)
            st.markdown('<div class="fg-section-label" style="font-size:0.9rem; margin-top:10px;">Top factors influencing this prediction:</div>', unsafe_allow_html=True)
            for rank, (feat, val) in enumerate(factors, start=1):
                label = FEATURE_LABELS.get(feat, feat)
                arrow, color = ("⬆", "#f87171") if val > 0 else ("⬇", "#4ade80")
                st.markdown(
                    f'<div class="fg-info-row"><span class="fg-info-label">{rank}. {label}</span>'
                    f'<span style="color:{color}; font-weight:700;">{arrow} {val:+.3f}</span></div>',
                    unsafe_allow_html=True,
                )
            method_note = "SHAP values (TreeExplainer)." if best_model_name in TREE_MODELS else "Linear contribution (coefficient × scaled value) — SHAP equivalent for a linear model."
            st.caption(method_note)

    st.write("")
    st.caption(
        "Educational project — trained on PaySim-schema synthetic data. "
        "Not intended for production use without retraining and validation on real-world data."
    )

# ---------------------------------------------------------------------------
# TAB 2 — Model Insights (figures déjà générées par src/train.py)
# ---------------------------------------------------------------------------
with tab_insights:
    figures = [
        ("05_model_comparison.png", "Model comparison — Precision / Recall / F1 / ROC-AUC"),
        ("07_confusion_matrices.png", "Confusion matrices"),
        ("08_feature_importance.png", "Feature importance (Random Forest & XGBoost)"),
        ("09_shap_summary.png", "SHAP summary plot (XGBoost)"),
    ]
    any_found = False
    for fname, caption in figures:
        fpath = os.path.join(FIG_DIR, fname)
        if os.path.exists(fpath):
            any_found = True
            with st.container(border=True):
                st.markdown(f'<div class="fg-section-label" style="font-size:0.95rem;">{caption}</div>', unsafe_allow_html=True)
                st.image(fpath, width="stretch")
            st.write("")
    if not any_found:
        st.info("No figures found. Run `python src/train.py` first to generate reports/figures/.")

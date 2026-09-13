"""
app.py — Interface interactive de détection de fraude
-------------------------------------------------------
Lancement :
    streamlit run app/app.py

L'utilisateur saisit les caractéristiques d'une transaction et obtient
une prédiction Fraude / Légitime avec la probabilité associée, produite
par le meilleur modèle entraîné (src/train.py).
"""

import os
import sys

import joblib
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessing import prepare_model_frame, single_transaction_to_frame  # noqa: E402

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

st.set_page_config(page_title="Détection de fraude", page_icon="🏦", layout="centered")


@st.cache_resource
def load_artifacts():
    with open(os.path.join(MODEL_DIR, "best_model.txt")) as f:
        best_model_name = f.read().strip()
    model_file = best_model_name.lower().replace(" ", "_") + ".joblib"
    model = joblib.load(os.path.join(MODEL_DIR, model_file))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.joblib"))
    return model, scaler, feature_names, best_model_name


st.title("🏦 Détection de fraude bancaire")
st.caption("Saisis les caractéristiques d'une transaction pour estimer si elle est frauduleuse.")

try:
    model, scaler, feature_names, best_model_name = load_artifacts()
except FileNotFoundError:
    st.error(
        "Aucun modèle entraîné trouvé. Lance d'abord :\n\n"
        "```bash\npython data/generate_sample_data.py\npython src/train.py\n```"
    )
    st.stop()

st.info(f"Modèle utilisé : **{best_model_name}**")

with st.form("transaction_form"):
    col1, col2 = st.columns(2)
    with col1:
        tx_type = st.selectbox("Type de transaction", ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"])
        amount = st.number_input("Montant", min_value=0.0, value=10000.0, step=100.0)
        step = st.number_input("Step (heure depuis le début, 1-744)", min_value=1, max_value=744, value=100)
        is_flagged = st.selectbox("Signalée par le système (isFlaggedFraud) ?", ["Non", "Oui"]) == "Oui"
    with col2:
        oldbalanceOrg = st.number_input("Solde émetteur avant transaction", min_value=0.0, value=10000.0, step=100.0)
        newbalanceOrig = st.number_input("Solde émetteur après transaction", min_value=0.0, value=0.0, step=100.0)
        oldbalanceDest = st.number_input("Solde destinataire avant transaction", min_value=0.0, value=0.0, step=100.0)
        newbalanceDest = st.number_input("Solde destinataire après transaction", min_value=0.0, value=10000.0, step=100.0)

    submitted = st.form_submit_button("Analyser la transaction", use_container_width=True)

if submitted:
    raw = single_transaction_to_frame(
        step=step,
        tx_type=tx_type,
        amount=amount,
        oldbalanceOrg=oldbalanceOrg,
        newbalanceOrig=newbalanceOrig,
        oldbalanceDest=oldbalanceDest,
        newbalanceDest=newbalanceDest,
        isFlaggedFraud=int(is_flagged),
    )
    X = prepare_model_frame(raw)
    X = X.reindex(columns=feature_names, fill_value=0)
    X_scaled = scaler.transform(X)

    proba_fraud = model.predict_proba(X_scaled)[0, 1]
    prediction = int(proba_fraud >= 0.5)

    st.divider()
    if prediction == 1:
        st.error(f"### 🚨 Transaction FRAUDULEUSE (probabilité : {proba_fraud:.1%})")
    else:
        st.success(f"### ✅ Transaction LÉGITIME (probabilité de fraude : {proba_fraud:.1%})")

    st.progress(min(float(proba_fraud), 1.0))
    st.caption(
        "Seuil de décision : 50%. Ajuste ce seuil dans le code (app/app.py) selon le "
        "compromis précision/recall souhaité en production."
    )

st.divider()
st.caption(
    "Modèle entraîné sur un jeu de données au schéma PaySim (voir data/generate_sample_data.py). "
    "Projet à but pédagogique — ne pas utiliser en production sans re-entraînement sur des données réelles."
)

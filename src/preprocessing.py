"""
preprocessing.py
-----------------
Fonctions de préparation des données, partagées entre le notebook,
le script d'entraînement (train.py) et l'application Streamlit.
"""

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",
]

TYPE_CATEGORIES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les variables dérivées utilisées pour l'EDA et enrichit le signal du modèle."""
    df = df.copy()
    df["balanceOrig_diff"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
    df["balanceDest_diff"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["errorBalanceOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    return df


def prepare_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit la matrice de features finale utilisée par les modèles :
    encodage one-hot de 'type' + variables d'erreur de solde (plus discriminantes
    et plus robustes qu'un simple LabelEncoder, tout en restant interprétables).
    """
    df = engineer_features(df)
    type_dummies = pd.get_dummies(df["type"], prefix="type")
    for cat in TYPE_CATEGORIES:
        col = f"type_{cat}"
        if col not in type_dummies.columns:
            type_dummies[col] = 0
    type_dummies = type_dummies[[f"type_{c}" for c in TYPE_CATEGORIES]]

    model_df = pd.concat(
        [
            df[
                [
                    "step",
                    "amount",
                    "oldbalanceOrg",
                    "newbalanceOrig",
                    "oldbalanceDest",
                    "newbalanceDest",
                    "isFlaggedFraud",
                    "errorBalanceOrig",
                    "errorBalanceDest",
                ]
            ],
            type_dummies,
        ],
        axis=1,
    )
    return model_df.astype(float)


def single_transaction_to_frame(
    step: int,
    tx_type: str,
    amount: float,
    oldbalanceOrg: float,
    newbalanceOrig: float,
    oldbalanceDest: float,
    newbalanceDest: float,
    isFlaggedFraud: int = 0,
) -> pd.DataFrame:
    """Construit un DataFrame à une ligne, au format brut, à partir des champs saisis dans l'UI."""
    return pd.DataFrame(
        [
            {
                "step": step,
                "type": tx_type,
                "amount": amount,
                "oldbalanceOrg": oldbalanceOrg,
                "newbalanceOrig": newbalanceOrig,
                "oldbalanceDest": oldbalanceDest,
                "newbalanceDest": newbalanceDest,
                "isFlaggedFraud": isFlaggedFraud,
            }
        ]
    )


# Libellés lisibles pour l'explication affichée dans l'app (section "Top factors")
FEATURE_LABELS = {
    "step": "moment de la transaction (step)",
    "amount": "montant de la transaction",
    "oldbalanceOrg": "solde émetteur avant transaction",
    "newbalanceOrig": "solde émetteur après transaction",
    "oldbalanceDest": "solde destinataire avant transaction",
    "newbalanceDest": "solde destinataire après transaction",
    "isFlaggedFraud": "signalement automatique du système",
    "errorBalanceOrig": "incohérence de solde côté émetteur",
    "errorBalanceDest": "incohérence de solde côté destinataire",
    "type_CASH_IN": "type de transaction (dépôt)",
    "type_CASH_OUT": "type de transaction (retrait)",
    "type_DEBIT": "type de transaction (débit)",
    "type_PAYMENT": "type de transaction (paiement)",
    "type_TRANSFER": "type de transaction (virement)",
}

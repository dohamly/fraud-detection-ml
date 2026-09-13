"""
generate_sample_data.py
------------------------
Génère un dataset synthétique respectant le schéma et les propriétés
statistiques du dataset "PaySim - Online Payments Fraud Detection" (Kaggle).

Pourquoi un dataset synthétique ?
Le dataset original (~470 Mo, 6.3M lignes) est trop volumineux pour être
téléchargé/versionné directement dans ce repo. Ce script produit un jeu de
données réaliste (mêmes colonnes, même logique métier, même déséquilibre de
classes ~0.1-0.3% de fraude, fraude concentrée sur TRANSFER/CASH_OUT) afin
que le pipeline (EDA -> preprocessing -> modélisation -> app) soit
entièrement reproductible.

Pour utiliser le vrai dataset Kaggle :
1. Télécharger "onlinefraud.csv" depuis
   https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset
2. Le placer dans data/online-payments_fraud.csv
3. Relancer src/train.py (aucune autre modification nécessaire)
"""

import numpy as np
import pandas as pd

RANDOM_STATE = 42
N_ROWS = 200_000
FRAUD_RATE = 0.0013  # ~0.13%, cohérent avec le dataset réel


def generate(n_rows: int = N_ROWS, fraud_rate: float = FRAUD_RATE) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)

    types = rng.choice(
        ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"],
        size=n_rows,
        p=[0.35, 0.34, 0.22, 0.08, 0.01],
    )

    step = rng.integers(1, 744, size=n_rows)  # 1 step = 1h sur 31 jours
    amount = np.round(rng.lognormal(mean=9.0, sigma=1.6, size=n_rows), 2)
    amount = np.clip(amount, 1, 9_000_000)

    old_orig = np.round(rng.lognormal(mean=9.5, sigma=1.8, size=n_rows), 2)
    old_orig = np.where(rng.random(n_rows) < 0.25, 0.0, old_orig)  # comptes vides fréquents

    old_dest = np.round(rng.lognormal(mean=9.0, sigma=1.9, size=n_rows), 2)
    old_dest = np.where(
        np.isin(types, ["PAYMENT", "DEBIT"]) & (rng.random(n_rows) < 0.7), 0.0, old_dest
    )

    # Fraude possible uniquement sur TRANSFER / CASH_OUT (comportement documenté PaySim)
    eligible = np.isin(types, ["TRANSFER", "CASH_OUT"])
    is_fraud = np.zeros(n_rows, dtype=int)
    n_target_fraud = int(n_rows * fraud_rate)
    eligible_idx = np.where(eligible)[0]
    fraud_idx = rng.choice(eligible_idx, size=min(n_target_fraud, len(eligible_idx)), replace=False)
    is_fraud[fraud_idx] = 1

    # Les transactions frauduleuses ciblent typiquement des montants élevés
    amount[fraud_idx] = np.round(rng.lognormal(mean=12.0, sigma=1.0, size=len(fraud_idx)), 2)
    old_orig[fraud_idx] = np.maximum(amount[fraud_idx] * rng.uniform(0.9, 1.5, len(fraud_idx)), amount[fraud_idx])

    # Soldes cohérents (avec un peu de bruit + incohérences injectées pour les fraudes,
    # comme observé empiriquement dans le vrai dataset)
    new_orig = np.maximum(old_orig - amount, 0)
    noise_mask = rng.random(n_rows) < 0.05
    new_orig[noise_mask] = old_orig[noise_mask]  # solde inchangé -> incohérence (signal utile)

    new_dest = old_dest + amount
    noise_mask_d = rng.random(n_rows) < 0.05
    new_dest[noise_mask_d] = old_dest[noise_mask_d]

    name_orig = np.array([f"C{rng.integers(1_000_000_000, 2_000_000_000)}" for _ in range(n_rows)])
    dest_prefix = np.where(np.isin(types, ["PAYMENT"]), "M", "C")
    name_dest = np.array(
        [f"{p}{rng.integers(1_000_000_000, 2_000_000_000)}" for p in dest_prefix]
    )

    is_flagged = np.zeros(n_rows, dtype=int)
    big_transfer = (types == "TRANSFER") & (amount > 200_000)
    is_flagged[big_transfer & (rng.random(n_rows) < 0.02)] = 1

    df = pd.DataFrame(
        {
            "step": step,
            "type": types,
            "amount": amount,
            "nameOrig": name_orig,
            "oldbalanceOrg": old_orig,
            "newbalanceOrig": new_orig,
            "nameDest": name_dest,
            "oldbalanceDest": old_dest,
            "newbalanceDest": new_dest,
            "isFraud": is_fraud,
            "isFlaggedFraud": is_flagged,
        }
    )
    return df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


if __name__ == "__main__":
    df = generate()
    out_path = "data/online-payments_fraud.csv"
    df.to_csv(out_path, index=False)
    print(f"Dataset généré : {out_path}")
    print(f"Lignes : {len(df):,} | Fraudes : {df['isFraud'].sum():,} ({df['isFraud'].mean()*100:.3f}%)")

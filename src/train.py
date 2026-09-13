"""
train.py
--------
Pipeline complet de détection de fraude :
EDA (figures exportées) -> preprocessing -> gestion du déséquilibre (SMOTE)
-> entraînement (Logistic Regression, Random Forest, XGBoost)
-> évaluation comparative (Precision/Recall/F1/ROC-AUC, matrices de confusion)
-> feature importance -> SHAP -> sauvegarde des artefacts (modèles, scaler, métriques).

Usage :
    python src/train.py
"""

import json
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from preprocessing import FEATURE_COLUMNS, prepare_model_frame

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_PATH = "data/online-payments_fraud.csv"
FIG_DIR = "reports/figures"
MODEL_DIR = "models"
sns.set_theme(style="whitegrid")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  -> figure sauvegardée : {path}")


# ---------------------------------------------------------------------------
# 1. Chargement des données
# ---------------------------------------------------------------------------
print("=" * 70)
print("1. CHARGEMENT DES DONNÉES")
print("=" * 70)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"{DATA_PATH} introuvable. Lance d'abord : python data/generate_sample_data.py "
        "(ou place le vrai dataset Kaggle à cet emplacement)."
    )

data = pd.read_csv(DATA_PATH)
print(f"Dimensions : {data.shape[0]:,} lignes x {data.shape[1]} colonnes")
print(f"Valeurs manquantes : {int(data.isnull().sum().sum())}")
print(f"Doublons : {int(data.duplicated().sum())}")

fraud_counts = data["isFraud"].value_counts()
fraud_pct = data["isFraud"].value_counts(normalize=True) * 100
print(f"\nDistribution de la cible :\n{fraud_counts}\n{fraud_pct.round(3)}")

# ---------------------------------------------------------------------------
# 2. EDA complète (figures exportées dans reports/figures)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("2. EDA")
print("=" * 70)

data_sample = data.sample(min(50_000, len(data)), random_state=RANDOM_STATE)

# 2.1 Distribution de la cible
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].bar(["Légitime", "Frauduleux"], fraud_counts.values, color=["#2ecc71", "#e74c3c"])
axes[0].set_title("Distribution des transactions")
axes[0].set_ylabel("Nombre")
axes[1].pie(
    fraud_counts.values,
    labels=["Légitime", "Frauduleux"],
    autopct="%1.3f%%",
    colors=["#2ecc71", "#e74c3c"],
    explode=(0, 0.15),
    shadow=True,
)
axes[1].set_title("Proportion")
savefig("01_target_distribution.png")

# 2.2 Fraude par type de transaction
fraud_by_type = data[data["isFraud"] == 1]["type"].value_counts()
fraud_rate_by_type = (fraud_by_type / data["type"].value_counts() * 100).sort_values(ascending=False)

fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
data_sample["type"].value_counts().plot(kind="bar", ax=axes[0], color="#3498db")
axes[0].set_title("Répartition des types de transaction")
fraud_by_type.plot(kind="bar", ax=axes[1], color="#e74c3c")
axes[1].set_title("Nombre de fraudes par type")
fraud_rate_by_type.plot(kind="bar", ax=axes[2], color="#f39c12")
axes[2].set_title("Taux de fraude par type (%)")
savefig("02_fraud_by_type.png")

# 2.3 Montants
fig, ax = plt.subplots(figsize=(9, 5))
sns.boxplot(x="isFraud", y="amount", data=data_sample, ax=ax)
ax.set_ylim(0, data_sample["amount"].quantile(0.99))
ax.set_title("Distribution des montants selon la fraude")
ax.set_xticklabels(["Légitime", "Frauduleux"])
savefig("03_amount_distribution.png")

# 2.4 Incohérences de solde (feature engineering signal)
tmp = data.copy()
tmp["balanceOrig_diff"] = tmp["newbalanceOrig"] - tmp["oldbalanceOrg"]
tmp["balanceDest_diff"] = tmp["newbalanceDest"] - tmp["oldbalanceDest"]
tmp["amount_check_orig"] = np.abs(tmp["balanceOrig_diff"] + tmp["amount"]) < 0.01
tmp["amount_check_dest"] = np.abs(tmp["balanceDest_diff"] - tmp["amount"]) < 0.01
suspicious = tmp[~tmp["amount_check_orig"] | ~tmp["amount_check_dest"]]
print(f"Taux de fraude parmi les transactions avec incohérence de solde : {suspicious['isFraud'].mean()*100:.2f}%")
print(f"Taux de fraude global : {data['isFraud'].mean()*100:.3f}%")

# 2.5 Matrice de corrélation
model_frame_full = prepare_model_frame(data)
corr_frame = model_frame_full.copy()
corr_frame["isFraud"] = data["isFraud"].values
corr = corr_frame.corr()
fig, ax = plt.subplots(figsize=(11, 8))
sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
ax.set_title("Matrice de corrélation (features encodées)")
savefig("04_correlation_matrix.png")
print("\nCorrélation avec isFraud :")
print(corr["isFraud"].sort_values(ascending=False))

# ---------------------------------------------------------------------------
# 3. Préparation des données
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("3. PRÉPARATION DES DONNÉES")
print("=" * 70)

X = prepare_model_frame(data)
y = data["isFraud"]
FEATURE_NAMES = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train : {X_train.shape} | Test : {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 4. Gestion du déséquilibre — SMOTE (appliqué uniquement sur le train)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("4. SMOTE (train uniquement, pour éviter toute fuite de données)")
print("=" * 70)

print(f"Avant SMOTE : {dict(pd.Series(y_train).value_counts())}")
smote = SMOTE(random_state=RANDOM_STATE)
X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)
print(f"Après SMOTE : {dict(pd.Series(y_train_sm).value_counts())}")

# ---------------------------------------------------------------------------
# 5. Entraînement des modèles
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("5. ENTRAÎNEMENT DES MODÈLES")
print("=" * 70)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}

results = {}
fitted_models = {}

for name, model in models.items():
    print(f"\n-> Entraînement : {name}")
    model.fit(X_train_sm, y_train_sm)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    fitted_models[name] = model
    print(f"   Precision={results[name]['precision']:.3f} | Recall={results[name]['recall']:.3f} "
          f"| F1={results[name]['f1']:.3f} | ROC-AUC={results[name]['roc_auc']:.3f}")

# ---------------------------------------------------------------------------
# 6. Comparaison des modèles
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("6. COMPARAISON DES MODÈLES")
print("=" * 70)

results_df = pd.DataFrame(results).T.round(4)
results_df = results_df[["accuracy", "precision", "recall", "f1", "roc_auc"]]
print(results_df)
results_df.to_csv(os.path.join("reports", "model_comparison.csv"))

fig, ax = plt.subplots(figsize=(10, 5.5))
results_df[["precision", "recall", "f1", "roc_auc"]].plot(kind="bar", ax=ax)
ax.set_title("Comparaison des modèles — Precision / Recall / F1 / ROC-AUC")
ax.set_ylabel("Score")
ax.set_xticklabels(results_df.index, rotation=0)
ax.legend(loc="lower right")
savefig("05_model_comparison.png")

# ROC curves superposées
fig, ax = plt.subplots(figsize=(7, 6))
for name, model in fitted_models.items():
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Courbes ROC — comparaison des modèles")
ax.legend(loc="lower right")
savefig("06_roc_curves_comparison.png")

# ---------------------------------------------------------------------------
# 7. Matrices de confusion (les 3 modèles)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("7. MATRICES DE CONFUSION")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, model) in zip(axes, fitted_models.items()):
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    ax.set_title(name)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
savefig("07_confusion_matrices.png")

# ---------------------------------------------------------------------------
# 8. Feature importance (Random Forest + XGBoost)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("8. FEATURE IMPORTANCE")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, name in zip(axes, ["Random Forest", "XGBoost"]):
    model = fitted_models[name]
    importances = pd.Series(model.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=True)
    importances.tail(12).plot(kind="barh", ax=ax, color="#2c3e50")
    ax.set_title(f"Feature importance — {name}")
savefig("08_feature_importance.png")

# ---------------------------------------------------------------------------
# 9. SHAP (XGBoost — modèle retenu)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("9. SHAP — INTERPRÉTABILITÉ (XGBoost)")
print("=" * 70)

best_model_name = results_df["f1"].idxmax()
print(f"Meilleur modèle (F1-score) : {best_model_name}")
xgb_model = fitted_models["XGBoost"]

sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_test_scaled), size=min(2000, len(X_test_scaled)), replace=False)
X_shap_sample = X_test_scaled[sample_idx]

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_shap_sample)

plt.figure(figsize=(9, 7))
shap.summary_plot(shap_values, X_shap_sample, feature_names=FEATURE_NAMES, show=False)
plt.title("SHAP summary plot — XGBoost")
savefig("09_shap_summary.png")

# ---------------------------------------------------------------------------
# 10. Sauvegarde des artefacts
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("10. SAUVEGARDE DES ARTEFACTS")
print("=" * 70)

for name, model in fitted_models.items():
    fname = name.lower().replace(" ", "_")
    joblib.dump(model, os.path.join(MODEL_DIR, f"{fname}.joblib"))

joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
joblib.dump(FEATURE_NAMES, os.path.join(MODEL_DIR, "feature_names.joblib"))

with open(os.path.join(MODEL_DIR, "best_model.txt"), "w") as f:
    f.write(best_model_name)

with open(os.path.join("reports", "metrics.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"Modèles sauvegardés dans {MODEL_DIR}/")
print(f"Meilleur modèle : {best_model_name}")
print("\nTerminé.")

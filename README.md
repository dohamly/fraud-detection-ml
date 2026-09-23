# FraudGuard AI — Backend + Frontend connectés

Cette version remplace la logique JS factice de la démo précédente par de vrais
appels à une API FastAPI qui charge votre modèle entraîné (XGBoost / SHAP).

## Structure attendue

Placez ces deux dossiers **à côté** de votre projet existant (celui qui contient
déjà `models/`, `reports/`, `src/preprocessing.py`, comme pour `app.py`) :

```
votre-projet/
  models/                  <- déjà existant
  reports/                 <- déjà existant
  src/preprocessing.py     <- déjà existant
  app.py                   <- votre app Streamlit (inchangée)
  backend/
    main.py                <- nouveau : API FastAPI
    requirements.txt
  frontend/
    index.html              <- nouveau : frontend HTML/JS connecté
```

Le backend importe `src/preprocessing.py` et lit les mêmes fichiers
(`models/*.joblib`, `reports/metrics.json`, `reports/figures/*.png`) que votre
`app.py`, avec exactement la même logique de préparation des features.

## Lancer le backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Vérifiez que ça tourne : ouvrez http://localhost:8000/docs (Swagger auto-généré).

## Lancer le frontend

Ouvrez simplement `frontend/index.html` dans votre navigateur (double-clic,
ou `python3 -m http.server` dans le dossier `frontend/` puis
http://localhost:8000... attention au port, utilisez un autre port que le
backend, ex. `python3 -m http.server 5500`).

Le frontend appelle `http://localhost:8000` par défaut — modifiable en tête du
`<script>` dans `index.html` (`const API_BASE = "..."`) si votre backend tourne
ailleurs.

## Endpoints de l'API

- `GET /health` — vérification rapide
- `GET /model-info` — nom du modèle, ROC-AUC, liste des figures disponibles
- `POST /predict` — reçoit une transaction, renvoie la probabilité de fraude
  et les facteurs SHAP les plus influents
- `GET /figures/<nom>.png` — sert les images de `reports/figures/`

## Ce qui change par rapport à Streamlit

- Aucun re-run complet du script à chaque clic : seul le résultat se met à jour.
- Toggle "No — Clear / Yes — Flagged" et jauge animés instantanément en JS.
- Le vrai modèle est appelé (plus de logique factice) — si le backend est
  éteint, l'interface affiche un message d'erreur clair au lieu de planter.

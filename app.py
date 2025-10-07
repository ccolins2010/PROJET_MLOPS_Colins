import streamlit as st
import pandas as pd
import joblib, json, sklearn
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="Risque crédit - PD", page_icon="📊", layout="centered")

# =========================================================
# 0) Entête pédagogique
# =========================================================
st.title("Évaluation du risque crédit : probabilité de défaut de paiement par client")
st.caption("Application académique : estimation de la probabilité de défaut à partir des caractéristiques du client.")

st.divider()

# =========================================================
# 1) Artefacts (modèle + métriques)
# =========================================================
ART = Path("artifacts")
MODEL_PATH = ART / "logistic_regression_final.joblib"    # <- meilleur modèle sélectionné en 4.6
METRICS_PATH = ART / "best_model_metrics.json"

# Charger métriques (si dispo)
metrics = {}
if METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text())
else:
    st.warning("Fichier des métriques introuvable : artifacts/best_model_metrics.json")

# Charger modèle avec cache
@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)

model = None
if MODEL_PATH.exists():
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Erreur au chargement du modèle : {e}")
else:
    st.error(f"Modèle introuvable : {MODEL_PATH}. Exécute la section 4.6 d’export dans le notebook.")

# Panneau d’infos techniques (repliable)
with st.expander("ℹ️ Informations modèle (cliquer pour afficher)"):
    st.markdown(f"**Modèle utilisé :** `{MODEL_PATH.name if MODEL_PATH.exists() else '—'}`")
    if metrics:
        st.markdown(
            f"- **ROC-AUC (test)** : {metrics.get('roc_auc', 0):.3f}  \n"
            f"- **PR-AUC (test)**  : {metrics.get('pr_auc', 0):.3f}  \n"
            f"- **Brier score**    : {metrics.get('brier', 0):.3f}  \n"
            f"- **Accuracy**       : {metrics.get('accuracy', 0):.3%}"
        )
    st.markdown(f"- **sklearn (inference)** : `{sklearn.__version__}`")
    st.markdown(f"- **Date d’exécution** : {datetime.now().strftime('%Y-%m-%d %H:%M')}")

st.divider()

# =========================================================
# 2) Aides / définitions
# =========================================================
st.markdown(
    "Analysez le profil de chaque client pour **anticiper le risque de défaut** et "
    "prendre des **décisions de crédit éclairées**."
)

st.divider()

# =========================================================
# 3) Fonctions utilitaires (verdict + recommandation + format €)
# =========================================================
def fmt_eur(x: float) -> str:
    """Format 10000 -> 10 000 €, 123456 -> 123 456 €."""
    return f"{x:,.0f} €".replace(",", " ").replace(".", ",")

def verdict_message(pd_value: float, threshold: float):
    """
    Retourne un niveau lisible + une recommandation métier.
    Les bornes sont indicatives (à adapter à vos politiques de risque).
    """
    if pd_value < 0.10:
        level = "Faible"
        advice = "Accepter le crédit (conditions standard)."
    elif pd_value < 0.30:
        level = "Modéré"
        advice = "Accepter sous conditions (limite d’endettement / stabilité des revenus)."
    elif pd_value < threshold:
        level = "Élevé (mais < seuil)"
        advice = "Étudier des garanties ou réduire le montant."
    else:
        level = "Très élevé (≥ seuil)"
        advice = "Demander des garanties fortes ou refuser."
    return level, advice

# =========================================================
# 4) Score unitaire (formulaire)
# =========================================================
st.subheader("Score unitaire")

col1, col2 = st.columns(2)
with col1:
    credit_lines_outstanding = st.number_input(
        "Lignes de crédit actives",
        min_value=0, step=1, value=1,
        help="Nombre de lignes de crédit actuellement ouvertes pour le client."
    )
    loan_amt_outstanding = st.number_input(
        "Montant du prêt en cours (€)",
        min_value=0.0, step=100.0, value=3000.0,
        help="Montant du prêt ciblé encore à rembourser."
    )
    total_debt_outstanding = st.number_input(
        "Dette totale (€)",
        min_value=0.0, step=100.0, value=8000.0,
        help="Somme de toutes les dettes du client (tous crédits confondus)."
    )
with col2:
    income = st.number_input(
        "Revenu annuel (€)",
        min_value=0.0, step=100.0, value=60000.0,
        help="Revenu brut annuel estimé du client."
    )
    years_employed = st.number_input(
        "Ancienneté (années)",
        min_value=0, step=1, value=5,
        help="Nombre d'années d'emploi du client."
    )
    fico_score = st.number_input(
        "Score FICO (300–850)",
        min_value=300, max_value=850, step=1, value=640,
        help="Indicateur de solvabilité (plus élevé = plus fiable)."
    )

theta = st.slider(
    "Seuil décisionnel (θ)",
    0.05, 0.95, 0.50, 0.01,
    help="Au-dessus du seuil → Risque élevé. En dessous → Fiable. Ajuster selon le coût FP/FN."
)

if st.button("⚙️ Prédire la probabilité de défaut (PD)"):
    if model is None:
        st.error("Modèle non chargé. Vérifie les artefacts ou les versions de librairies.")
    else:
        X = pd.DataFrame([{
            "credit_lines_outstanding": credit_lines_outstanding,
            "loan_amt_outstanding": loan_amt_outstanding,
            "total_debt_outstanding": total_debt_outstanding,
            "income": income,
            "years_employed": years_employed,
            "fico_score": fico_score
        }])
        pd_hat = float(model.predict_proba(X)[:, 1])
        verdict_is_risk = (pd_hat >= theta)

        # Indicateur principal
        st.metric("Probabilité de défaut (PD)", f"{pd_hat:.2%}")

        # Niveau + recommandation (plus parlant)
        niveau, reco = verdict_message(pd_hat, theta)

        # Badge coloré + phrase complète
        if verdict_is_risk:
            st.markdown(
                "<div style='padding:10px;border-radius:8px;background:#ffe6e6;color:#a80000;font-weight:600;'>"
                f"❌ Verdict : Risque élevé — PD = {pd_hat:.2%} (niveau : {niveau}). "
                f"**Décision suggérée : {reco}**"
                "</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div style='padding:10px;border-radius:8px;background:#e9f7ef;color:#1e7e34;font-weight:600;'>"
                f"✅ Verdict : Fiable — PD = {pd_hat:.2%} (niveau : {niveau}). "
                f"**Décision suggérée : {reco}**"
                "</div>",
                unsafe_allow_html=True
            )

        # Rappel des entrées (lisible)
        with st.expander("Détails de la saisie"):
            st.write({
                "Lignes de crédit actives": credit_lines_outstanding,
                "Montant du prêt en cours": fmt_eur(loan_amt_outstanding),
                "Dette totale": fmt_eur(total_debt_outstanding),
                "Revenu annuel": fmt_eur(income),
                "Ancienneté (années)": years_employed,
                "Score FICO": fico_score,
                "Seuil (θ)": round(theta, 2),
            })

st.divider()

# =========================================================
# 5) Scoring par lot (CSV) — version avec "Comment ça marche ?"
# =========================================================
st.subheader("Scoring par lot (CSV)")

st.markdown(
    "#### Comment ça marche ?\n"
    "1. **Téléchargez le modèle de CSV** ci-dessous et remplissez-le avec vos clients.\n"
    "2. **Importez** le fichier via le bouton.\n"
    "3. L’app calcule **PD** + **verdict** pour chaque ligne.\n"
    "4. **Téléchargez** le fichier enrichi pour vos décisions."
)

# Option : colonnes explicatives (désactivé par défaut)
show_explanatory = st.checkbox(
    "Ajouter des colonnes explicatives (niveau de risque, recommandation)",
    value=False
)

# CSV modèle téléchargeable
sample = pd.DataFrame([{
    "credit_lines_outstanding": 1,
    "loan_amt_outstanding": 3500,
    "total_debt_outstanding": 9000,
    "income": 65000,
    "years_employed": 4,
    "fico_score": 630
}])
st.download_button(
    "⬇️ Télécharger un CSV modèle",
    sample.to_csv(index=False).encode("utf-8"),
    "modele_clients.csv",
    "text/csv"
)

# Uploader
file = st.file_uploader("Importer un CSV", type=["csv"])
required = [
    "credit_lines_outstanding",
    "loan_amt_outstanding",
    "total_debt_outstanding",
    "income",
    "years_employed",
    "fico_score",
]

if file:
    if model is None:
        st.error("Modèle non chargé. Impossible de scorer le CSV.")
    else:
        try:
            df = pd.read_csv(file)
        except Exception as e:
            st.error(f"Erreur de lecture du CSV : {e}")
            df = None

        if df is not None:
            missing = [c for c in required if c not in df.columns]
            if missing:
                st.error(f"Colonnes manquantes : {missing}")
            else:
                out = df.copy()
                # Colonnes de base (applique aussi le seuil θ choisi dans l'UI)
                out["pd"] = model.predict_proba(out[required])[:, 1]
                out["verdict"] = (out["pd"] >= theta).map({True: "Risque élevé", False: "Fiable"})

                # Colonnes explicatives optionnelles (lisibilité métier)
                if show_explanatory:
                    niveaux, recos = [], []
                    for p in out["pd"].tolist():
                        lvl, rc = verdict_message(p, theta)  # réutilise ta fonction définie plus haut
                        niveaux.append(lvl)
                        recos.append(rc)
                    out["niveau_risque"] = niveaux
                    out["recommandation"] = recos

                st.dataframe(out.head(20), use_container_width=True)
                st.download_button(
                    "💾 Télécharger les résultats (CSV)",
                    out.to_csv(index=False).encode("utf-8"),
                    "predictions.csv",
                    "text/csv"
                )



# =========================================================
# 6) Pied de page
# =========================================================
st.divider()
st.markdown("👤 **Auteur :** Colins DONGMO — Master Data Science")

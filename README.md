# Évaluation du risque crédit – Probabilité de défaut (PD)

![CI/CD – Deploy Streamlit to ECS (Paris)](https://github.com/ccolins2010/PROJET_MLOPS_Colins/actions/workflows/aws.yaml/badge.svg)

Application académique permettant d’estimer la **probabilité de défaut (PD)** d’un client à partir de ses caractéristiques, avec **verdict** (Fiable / Risque élevé) et **recommandation** métier.  
Projet mené de bout-en-bout (prétraitement, modèles, export d’artefacts, app Streamlit, CI/CD vers **AWS ECS Fargate**).

---

## 🎯 Objectifs

- Construire un **modèle de classification** (probabilité de défaut).
- Comparer plusieurs algorithmes, sélectionner le meilleur.
- Exposer une **application Streamlit** pour scoring unitaire + batch CSV.
- **Automatiser le déploiement** via GitHub Actions → ECR/ECS (Paris eu-west-3).

---

## 🧰 Stack technique

- **Python** · Pandas · NumPy · scikit-learn
- **Streamlit** (UI)
- **MLflow** (recommandé) pour le suivi d’expériences
- **Docker** + **AWS ECR/ECS (Fargate)** pour le déploiement
- **GitHub Actions** pour la CI/CD

---

## 📦 Données

- Fichier : `Data/Loan_Data.csv`  
- **Cible** : `default` (0 = non-défaut, 1 = défaut)

| Variable                    | Description                                      |
|----------------------------|--------------------------------------------------|
| `credit_lines_outstanding` | Lignes de crédit actives                         |
| `loan_amt_outstanding`     | Montant du prêt en cours                         |
| `total_debt_outstanding`   | Dette totale (tous crédits)                      |
| `income`                   | Revenu annuel                                    |
| `years_employed`           | Ancienneté (années)                              |
| `fico_score`               | Score FICO (300–850, plus élevé = plus fiable)   |
| `default`                  | **Cible** (0/1)                                  |

---

## 🔬 Méthodologie (résumé)

1. **EDA & Pré-traitement** : valeurs manquantes, distributions, standardisation (via `Pipeline`).
2. **Model Engineering** : Logistic Regression (balanced), Decision Tree, Random Forest.
3. **Évaluation** : ROC-AUC, PR-AUC, Brier Score, Accuracy, matrice de confusion.  
   → **Sélection** par PR-AUC puis ROC-AUC.
4. **Export artefacts** : modèle + métriques dans `artifacts/`.
5. **App Streamlit** : scoring unitaire + **batch CSV**.

---

## 🖥️ Application Streamlit

- **Score unitaire** (formulaire) → PD + verdict + recommandation.
- **Seuil (θ)** ajustable (compromis faux positifs / faux négatifs).
- **Scoring par lot** : upload CSV → ajout des colonnes `pd`, `verdict` (+ option : niveau & reco).

---

## 🚀 Lancer en local

**Pré-requis** : Python 3.10+

```bash
git clone https://github.com/ccolins2010/PROJET_MLOPS_Colins.git
cd PROJET_MLOPS_Colins
pip install -r requirements.txt
streamlit run app.py

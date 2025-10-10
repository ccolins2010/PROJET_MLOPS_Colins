# Évaluation du risque crédit – Probabilité de défaut (PD)

![CI/CD](https://github.com/ccolins2010/PROJET_MLOPS_Colins/actions/workflows/github-docker-cicd.yaml/badge.svg)

![banner](banner.jpg)

## Démo en ligne
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://projetmlopscolins-ppdqiepfahsufazyhn4ngk.streamlit.app/)

---

## 🎯 Description
Projet MLOps end-to-end : à partir des caractéristiques d’un client, l’application **estime la probabilité de défaut (PD)**, affiche un **verdict** (Fiable / Risque élevé) et une **recommandation**.  
Trois modèles ont été entraînés et comparés ; le **meilleur** est **exporté** puis **déployé** sur **Streamlit Cloud**.

- **Meilleur modèle** : Régression Logistique  
- **Artefacts exportés** :
  - `artifacts/logistic_regression_final.joblib`
  - `artifacts/best_model_metrics.json`
  - `artifacts/comparaison_modeles.csv`
  - `artifacts/sklearn_version.txt`

---

## 🧰 Technologies
Python • Pandas • NumPy • scikit-learn • Matplotlib/Seaborn • Streamlit • *(MLflow recommandé)*  
Déploiement : Streamlit Community Cloud (+ CI/CD GitHub Actions → Docker Hub et AWS ECS)

---

## 🔬 Méthodologie (résumé)
1. **EDA & Pré-traitement** : inspection, valeurs manquantes, distributions, corrélations, imputation/standardisation via `Pipeline`.
2. **Model Engineering** :
   - Logistic Regression (class_weight="balanced")
   - Decision Tree
   - Random Forest
3. **Évaluation** : ROC-AUC, PR-AUC, Brier score, Accuracy, matrice de confusion.  
   **Sélection** par PR-AUC puis ROC-AUC.
4. **Export** du pipeline entraîné + métriques → `artifacts/`.
5. **Déploiement** Streamlit : scoring unitaire et **batch CSV**.

---

## 📦 Données
- Fichier : `Data/Loan_Data.csv`  
- **Cible** : `default` (0 = non-défaut, 1 = défaut)

| Variable                    | Description                                    |
|----------------------------|-----------------------------------------------|
| `credit_lines_outstanding` | Lignes de crédit actives                      |
| `loan_amt_outstanding`     | Montant du prêt en cours                      |
| `total_debt_outstanding`   | Dette totale (tous crédits)                   |
| `income`                   | Revenu annuel                                  |
| `years_employed`           | Ancienneté (années)                            |
| `fico_score`               | Score FICO (300–850, plus élevé = plus fiable) |
| `default`                  | **Cible** (0/1)                                |

---

## 🖥️ Application Streamlit
- **Score unitaire** : formulaire → PD + verdict + recommandation.  
- **Seuil (θ)** ajustable (compromis faux positifs / faux négatifs).  
- **Scoring par lot (CSV)** : upload → ajout des colonnes `pd`, `verdict` → download.  
- L’app charge automatiquement les artefacts depuis `artifacts/`.

---

## 🧪 Suivi d’expériences (MLflow – recommandé)
- 1 **experiment** par modèle ; chaque itération = **run** (métriques/params/artefacts).  
- Dossiers locaux : `mlruns/` et `mlartifacts/`.  
- Prévoir des **captures MLflow UI** pour la soutenance.

---

## 🚀 Exécution locale

**Pré-requis** : Python **3.10** (recommandé pour compatibilité `sklearn`/`numpy`).

```bash
git clone https://github.com/ccolins2010/PROJET_MLOPS_Colins.git
cd PROJET_MLOPS_Colins
pip install -r requirements.txt
streamlit run app.py

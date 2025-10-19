# 💳 Évaluation du risque crédit – Probabilité de défaut (PD)

![CI/CD – Deploy Streamlit to ECS (Paris)](https://github.com/ccolins2010/PROJET_MLOPS_Colins/actions/workflows/aws.yaml/badge.svg)

Application académique permettant d’estimer la **probabilité de défaut (PD)** d’un client à partir de ses caractéristiques, avec **verdict** (Fiable / Risque élevé) et **recommandation** métier.  
Projet mené **de bout en bout (MLOps complet)** : prétraitement, modèles, tracking MLflow, export d’artefacts, application Streamlit et **déploiement automatisé sur AWS ECS (Fargate)** via **GitHub Actions**.

---

## 🎯 Objectifs

- Construire un **modèle de classification** pour prédire la probabilité de défaut de crédit.  
- Comparer plusieurs algorithmes de machine learning.  
- Traquer les expérimentations avec **MLflow** (un modèle = un experiment).  
- Sélectionner le meilleur modèle et le déployer dans une **app Streamlit**.  
- Mettre en place une **pipeline CI/CD complète** pour automatiser le déploiement cloud.

---

## 🧰 Stack technique

- **Python 3.10+**
- **Pandas**, **NumPy**, **scikit-learn**
- **Streamlit** (interface utilisateur)
- **MLflow** (tracking et gestion des modèles)
- **Docker** + **AWS ECS Fargate** (déploiement)
- **GitHub Actions** (CI/CD automatisée)

---

## 📦 Données

- Fichier source : `Data/Loan_Data.csv`  
- **Cible** : `default` (0 = non défaut, 1 = défaut)

| Variable                    | Description                                      |
|-----------------------------|--------------------------------------------------|
| `credit_lines_outstanding`  | Lignes de crédit actives                         |
| `loan_amt_outstanding`      | Montant du prêt en cours (€)                     |
| `total_debt_outstanding`    | Dette totale (€)                                 |
| `income`                    | Revenu annuel (€)                                |
| `years_employed`            | Ancienneté (années)                              |
| `fico_score`                | Score FICO (300–850, plus élevé = plus fiable)   |
| `default`                   | **Cible** (0/1)                                  |

---

## 🔬 Méthodologie

1. **EDA & Pré-traitement**  
   - Nettoyage, imputation médiane, standardisation.  
   - Création d’un `Pipeline` scikit-learn reproductible.

2. **Model Engineering**  
   - 3 algorithmes testés :  
     - **Logistic Regression**  
     - **Decision Tree**  
     - **Random Forest**  
   - Chaque modèle = **expérience MLflow**.  
   - Chaque variation d’hyperparamètres = **run MLflow**.

3. **Tracking MLflow (expériences & runs)**  
   - **Params loggés** : hyperparamètres, seuil, features.  
   - **Metrics loggées** : ROC-AUC, PR-AUC, Brier score, Accuracy.  
   - **Artifacts loggés** : courbes ROC, PR, matrices de confusion.  
   - **Traçabilité complète** :  
     - Dataset SHA-256 + chemin  
     - Graine de split (reproductibilité)  
   - Sélection automatique du meilleur modèle via `mlflow.search_runs()`.  
   - Rechargement du modèle gagnant par URI `runs:/<run_id>/model`.

4. **Export des artefacts**  
   - Dossier `artifacts/` contenant :  
     - `best_model_from_mlflow_*.joblib`  
     - `best_model_metrics.json`  
     - Courbes et matrices PNG.

5. **App Streamlit**  
   - Scoring **unitaire** : saisie manuelle du profil client.  
   - Scoring **par lot** : upload CSV → calcul PD + verdict pour chaque client.  
   - **Seuil décisionnel (θ)** ajustable pour simuler les stratégies de risque.

---

## 🖥️ Application Streamlit

Fonctionnalités :
- Affichage du **modèle chargé**, des **métriques de test** et de la **version sklearn**.
- **Score unitaire** : saisie des variables, affichage de la probabilité de défaut (PD) et du verdict.
- **Score par lot (CSV)** : import d’un fichier clients → prédiction PD + verdict + téléchargement CSV enrichi.
- **Seuil (θ)** modulable pour ajuster la décision métier.

> Exemple d’interface :  
> - 🟢 PD faible → « Accepter le crédit (conditions standard) »  
> - 🔴 PD élevé → « Refuser ou demander garanties ».

---

## ⚙️ Lancer en local

**Pré-requis** : Python 3.10+

```bash
git clone https://github.com/ccolins2010/PROJET_MLOPS_Colins.git
cd PROJET_MLOPS_Colins
pip install -r requirements.txt
streamlit run app.py

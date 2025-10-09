# ---------- Dockerfile (recommandé) ----------
FROM python:3.10-slim

# Bonnes pratiques Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1) Installer les dépendances (cache-friendly)
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 2) Copier le code et les artefacts (modèle .joblib, métriques…)
COPY . .

# 3) Exposer Streamlit et lancer l'app
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
# --------------------------------------------

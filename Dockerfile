# Utilise une image légère de Python
FROM python:3.9-slim

# Éviter les prompts
ENV DEBIAN_FRONTEND=noninteractive

# Installer dépendances système utiles (pour OpenCV et autres)
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Définir le dossier de travail
WORKDIR /app

# Copier requirements.txt et l’installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source (exclure les dossiers volumineux comme uploads/ et weights/ avec .dockerignore)
COPY . .

# Exposer le port utilisé par Flask (par défaut 5000, mais ajustable via Koyeb)
EXPOSE 5000

# Commande de lancement avec gunicorn
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "routes:app"]

# Utilise une image légère de Python
FROM python:3.9-slim

# Éviter les prompts
ENV DEBIAN_FRONTEND=noninteractive

# Installer dépendances système utiles (si OpenCV, PIL, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Créer le dossier de travail
WORKDIR /app

# Copier requirements.txt et l’installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code source
COPY . .

# Exposer le port utilisé par Flask
EXPOSE 5000

# Commande de lancement
CMD ["python", "main.py"]

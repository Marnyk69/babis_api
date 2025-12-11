FROM python:3.10-slim

# -------------------------------
# Systémové závislosti
# -------------------------------
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------
# Nastavení pracovního adresáře
# -------------------------------
WORKDIR /app

# -------------------------------
# Zkopíruj requirements
# -------------------------------
COPY requirements.txt .

# -------------------------------
# Instalace Python knihoven
# (Torch CPU, Transformers, SentenceTransformers…)
# -------------------------------
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------
# Zkopíruj model (pokud je uložen lokálně)
# -------------------------------
COPY robeczech_babis_freeze_last2 ./robeczech_babis_freeze_last2

# -------------------------------
# Zkopíruj CSV dataset (volitelné)
# -------------------------------
COPY politici_vyroky.csv .

# -------------------------------
# Zkopíruj aplikaci
# -------------------------------
COPY main.py .

# -------------------------------
# Spusť FastAPI přes uvicorn
# -------------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

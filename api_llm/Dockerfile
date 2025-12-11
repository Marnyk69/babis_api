FROM python:3.10-slim

# Optimalizace: nainstalujeme system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Nastavíme pracovní adresář
WORKDIR /app

# Zkopírujeme requirements
COPY requirements.txt .

# Instalace Python knihoven
RUN pip install --no-cache-dir -r requirements.txt

# Zkopíruj aplikaci
COPY main.py .

# Uvicorn – spustíme API server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


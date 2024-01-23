FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 

WORKDIR /app

COPY requirements.txt .
COPY api.py .
COPY load_model.py .
COPY segtools ./segtools
COPY VOCdevkit ./VOCdevkit
COPY models ./models
COPY Autoencoder.py .

RUN apt-get update && \
    apt-get install -y python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "api:app", "--bind", "0.0.0.0:8000", "--timeout", "1000"]

# docker build -t mlops-backend .
# docker run --gpus all -p 8000:8000 --name mlops-container -t mlops-backend

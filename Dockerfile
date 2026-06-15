# ============================================================
#  Dockerfile — MLOps Enfermedad ONNX · Proyecto Final
#  Universidad Icesi · Maestría en IA Aplicada
#
#  Compatible con Google Cloud Run:
#    - Escucha en $PORT (default 8080)
#    - Imagen ligera basada en python:3.11-slim
#    - El modelo ONNX se copia durante el build
# ============================================================

FROM python:3.11-slim

LABEL maintainer="Universidad Icesi - MLOps"
LABEL description="API ONNX de predicción de enfermedad"
LABEL version="1.0.0"

# Evitar generación de archivos .pyc y forzar stdout sin buffer
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Dependencias del sistema (mínimas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencias de Python (capa cacheada)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --trusted-host pypi.org \
    --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org \
    -r requirements.txt

# Copiar código de la aplicación
COPY app/ ./app/

# Copiar modelo ONNX (debe estar descargado antes del build)
# Si MODEL_PATH se sobrescribe en runtime, puede apuntar a otra ruta.
COPY models/modelo_enfermedad.onnx /app/models/modelo_enfermedad.onnx

# Variable por defecto (puede sobrescribirse en Cloud Run)
ENV MODEL_PATH=/app/models/modelo_enfermedad.onnx

# Exponer puerto
EXPOSE 8080

# Health check para Cloud Run
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Iniciar la aplicación. Cloud Run inyecta $PORT automáticamente.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}

"""
API FastAPI — Predicción de Estado de Enfermedad con ONNX.

Universidad Icesi · Maestría en IA Aplicada · Proyecto Final MLOps

Endpoints:
    GET  /            → Página de bienvenida (HTML simple)
    GET  /health      → Health check para Cloud Run
    GET  /model-info  → Información del modelo cargado
    POST /predict     → Predicción de estado de enfermedad
"""

import os
import uuid
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from app.schemas import (
    PatientInput,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)
from app.inference import predict, get_model, CLASS_LABELS, FEATURE_ORDER
from app.prediction_logger import get_logger

# Configuración de logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# Variables de entorno
ENVIRONMENT = os.environ.get("ENVIRONMENT", "local")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1.0.0")
MODEL_PATH = os.environ.get("MODEL_PATH", "models/modelo_enfermedad.onnx")


# ─────────────────────────────────────────────
# Ciclo de vida (carga del modelo al inicio)
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo y verifica el logger al iniciar."""
    logger.info(f"Iniciando aplicación en entorno: {ENVIRONMENT}")
    logger.info(f"Versión del modelo: {MODEL_VERSION}")
    logger.info(f"Ruta del modelo: {MODEL_PATH}")

    try:
        get_model()
        logger.info("✅ Modelo ONNX cargado correctamente")
    except Exception as e:
        logger.error(f"❌ Error cargando modelo: {e}")

    get_logger()
    logger.info("✅ Logger de predicciones inicializado")
    yield
    # Cleanup (si fuera necesario) iría aquí


# Inicializar FastAPI
app = FastAPI(
    title="MLOps Enfermedad — ONNX",
    description=(
        "API de predicción de estado de enfermedad usando ONNX Runtime.  \n"
        "**Universidad Icesi — Proyecto Final MLOps**"
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home():
    """Página de bienvenida simple."""
    return f"""
    <html>
    <head><title>MLOps Enfermedad — {ENVIRONMENT.upper()}</title></head>
    <body style="font-family: 'Segoe UI', sans-serif; padding: 40px; max-width: 700px; margin: auto;">
        <h1 style="color: #003366;">MLOps · Predicción de Enfermedad</h1>
        <p>Servicio de inferencia con modelo ONNX.</p>
        <ul>
            <li><b>Entorno:</b> {ENVIRONMENT}</li>
            <li><b>Versión del modelo:</b> {MODEL_VERSION}</li>
            <li><b>Documentación:</b> <a href="/docs">/docs</a></li>
            <li><b>Health:</b> <a href="/health">/health</a></li>
            <li><b>Info del modelo:</b> <a href="/model-info">/model-info</a></li>
        </ul>
        <hr/>
        <p style="color:#888; font-size:.85rem;">
            Universidad Icesi · Maestría en IA Aplicada
        </p>
    </body>
    </html>
    """


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — usado por Cloud Run para verificar disponibilidad."""
    model_loaded = False
    try:
        get_model()
        model_loaded = True
    except Exception as e:
        logger.warning(f"Health check: modelo no cargado: {e}")

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        environment=ENVIRONMENT,
        model_loaded=model_loaded,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Retorna información del modelo cargado."""
    return ModelInfoResponse(
        model_version=MODEL_VERSION,
        model_path=MODEL_PATH,
        input_features=FEATURE_ORDER,
        output_classes=list(CLASS_LABELS.values()),
        environment=ENVIRONMENT,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(patient: PatientInput):
    """
    Predice el estado de enfermedad de un paciente.

    Cada predicción se registra en GCS (o archivo local como fallback).
    """
    try:
        input_dict = patient.model_dump()
        result = predict(input_dict)

        prediction_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        response = PredictionResponse(
            environment=ENVIRONMENT,
            prediction=result["prediction_label"],
            prediction_class=result["prediction_class"],
            prediction_id=prediction_id,
            model_version=MODEL_VERSION,
            timestamp=timestamp,
            input=input_dict,
        )

        # Registrar predicción
        try:
            get_logger().log_prediction(response.model_dump())
        except Exception as e:
            logger.warning(f"No se pudo registrar predicción: {e}")

        return response

    except FileNotFoundError as e:
        logger.error(f"Modelo no disponible: {e}")
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

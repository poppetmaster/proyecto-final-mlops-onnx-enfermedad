"""
API FastAPI — Predicción de Estado de Enfermedad con ONNX.

Universidad Icesi · Maestría en IA Aplicada · Proyecto Final MLOps

Endpoints:
    GET  /            → Formulario web para el médico (HTML)
    GET  /health      → Health check para Cloud Run
    GET  /model-info  → Información del modelo cargado
    POST /predict     → Predicción vía API REST (JSON)
    POST /predecir    → Predicción vía formulario web (HTML)
"""

import os
import uuid
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
from jinja2 import Template

from app.schemas import (
    PatientInput,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)
from app.inference import predict, get_model, CLASS_LABELS, FEATURE_ORDER
from app.prediction_logger import get_logger

# ─────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ENVIRONMENT   = os.environ.get("ENVIRONMENT",   "local")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1.0.0")
MODEL_PATH    = os.environ.get("MODEL_PATH",    "models/modelo_enfermedad.onnx")


# ─────────────────────────────────────────────
# Ciclo de vida
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
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


# ─────────────────────────────────────────────
# Aplicación FastAPI
# ─────────────────────────────────────────────

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
# Template HTML — formulario + resultado
# ─────────────────────────────────────────────

PAGINA_HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Predicción de Enfermedad — Icesi {{ env_badge }}</title>
<style>
  :root {
    --azul:    #003366;
    --acento:  #2E86AB;
    --verde:   #27ae60;
    --amarillo:#f39c12;
    --naranja: #e67e22;
    --rojo:    #c0392b;
    --negro:   #2c2c2c;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #f0f4f8; color: #333;
    min-height: 100vh; display: flex; flex-direction: column;
    align-items: center; padding: 30px 16px;
  }

  /* ── header ── */
  .header { text-align: center; margin-bottom: 28px; }
  .header h1 { color: var(--azul); font-size: 1.6rem; margin-bottom: 6px; }
  .header p  { color: #666; font-size: .9rem; }
  .env-tag {
    display: inline-block; margin-top: 8px;
    padding: 3px 14px; border-radius: 20px; font-size: .78rem;
    font-weight: 700; letter-spacing: .5px; text-transform: uppercase;
    background: {{ "#1a7a4a" if environment == "prod" else "#1565C0" }};
    color: #fff;
  }

  /* ── card ── */
  .card {
    background: #fff; border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0,0,0,.08);
    width: 100%; max-width: 520px; padding: 32px; margin-bottom: 24px;
  }
  .card h2 {
    color: var(--azul); font-size: 1.15rem;
    margin-bottom: 20px; border-bottom: 2px solid var(--acento);
    padding-bottom: 8px;
  }

  /* ── form ── */
  label {
    display: block; font-weight: 600; font-size: .85rem;
    color: #444; margin-bottom: 4px; margin-top: 14px;
  }
  input[type=number] {
    width: 100%; padding: 10px 12px; border: 1px solid #ccd;
    border-radius: 8px; font-size: .95rem; transition: border .2s;
  }
  input[type=number]:focus { border-color: var(--acento); outline: none; }
  .checks { display: flex; gap: 24px; margin-top: 16px; }
  .checks label {
    display: flex; align-items: center; gap: 6px;
    font-weight: 500; cursor: pointer;
  }
  .checks input[type=checkbox] {
    width: 18px; height: 18px; accent-color: var(--acento);
  }
  button {
    margin-top: 24px; width: 100%; padding: 13px;
    background: var(--azul); color: #fff; border: none;
    border-radius: 8px; font-size: 1rem; font-weight: 600;
    cursor: pointer; transition: background .2s;
  }
  button:hover { background: var(--acento); }
  .nav-links { margin-top: 14px; text-align: center; font-size: .85rem; }
  .nav-links a {
    color: var(--acento); text-decoration: none; margin: 0 8px;
  }
  .nav-links a:hover { text-decoration: underline; }

  /* ── resultado ── */
  .resultado { text-align: center; padding: 24px; }
  .resultado h2 { margin-bottom: 18px; }
  .badge {
    display: inline-block; padding: 10px 28px; border-radius: 30px;
    font-size: 1.15rem; font-weight: 700; color: #fff; margin-bottom: 12px;
  }
  .badge.no-enfermo { background: var(--verde);    }
  .badge.leve       { background: var(--amarillo); }
  .badge.aguda      { background: var(--naranja);  }
  .badge.cronica    { background: var(--rojo);     }
  .badge.terminal   { background: var(--negro);    }
  .score { font-size: 2rem; font-weight: 700; color: var(--azul); }
  .desc  { margin-top: 10px; font-size: .9rem; color: #555; line-height: 1.5; }
  .meta  {
    margin-top: 16px; font-size: .78rem; color: #888;
    background: #f8f9fa; border-radius: 8px; padding: 12px; text-align: left;
  }
  .meta span { display: inline-block; margin: 2px 4px; }

  /* ── params ── */
  .params {
    margin-top: 12px; font-size: .8rem; color: #888;
    background: #f0f4f8; border-radius: 8px; padding: 12px; text-align: left;
  }

  /* ── footer ── */
  footer {
    margin-top: auto; padding-top: 20px;
    font-size: .75rem; color: #aaa; text-align: center;
  }
  .aviso {
    margin-top: 14px; font-size: .78rem; color: #999;
    text-align: center; font-style: italic;
  }
</style>
</head>
<body>

<div class="header">
  <h1>Predicción de Estado de Enfermedad modificado</h1>
  <p>Universidad Icesi · Pipeline de MLOps · Proyecto Final</p>
  <span class="env-tag">{{ environment }} · {{ model_version }}</span>
</div>

<!-- ── FORMULARIO ── -->
<div class="card">
  <h2>Datos del Paciente</h2>
  <form method="post" action="/predecir">

    <label for="edad">Edad del paciente</label>
    <input type="number" id="edad" name="edad"
           min="0" max="120" value="{{ edad }}"
           required placeholder="Ej: 45"/>

    <label for="num_sintomas">Número de síntomas reportados (0-20)</label>
    <input type="number" id="num_sintomas" name="num_sintomas"
           min="0" max="20" value="{{ num_sintomas }}"
           required placeholder="Ej: 5"/>

    <label for="dias_sintomas">Días con síntomas activos (0-365)</label>
    <input type="number" id="dias_sintomas" name="dias_sintomas"
           min="0" max="365" value="{{ dias_sintomas }}"
           required placeholder="Ej: 7"/>

    <label for="nivel_dolor">Nivel de dolor (0-10)</label>
    <input type="number" id="nivel_dolor" name="nivel_dolor"
           min="0" max="10" value="{{ nivel_dolor }}"
           required placeholder="Ej: 6"/>

    <div class="checks">
      <label>
        <input type="checkbox" name="tiene_fiebre"
               {{ "checked" if tiene_fiebre else "" }}/> Tiene fiebre
      </label>
      <label>
        <input type="checkbox" name="enfermedad_base"
               {{ "checked" if enfermedad_base else "" }}/> Enfermedad de base
      </label>
    </div>

    <button type="submit">Obtener Predicción</button>
  </form>

  <div class="nav-links">
    <a href="/health">🟢 Health</a>
    <a href="/model-info">🧠 Modelo</a>
    <a href="/docs">📄 API Docs</a>
  </div>
  <p class="aviso">
    Herramienta de apoyo académico. No reemplaza el diagnóstico médico profesional.
  </p>
</div>

<!-- ── RESULTADO ── -->
{% if resultado %}
<div class="card resultado">
  <h2>Resultado de la Predicción</h2>

  {% set cls = "no-enfermo" if resultado.prediction == "NO ENFERMO"
          else "leve"       if resultado.prediction == "ENFERMEDAD LEVE"
          else "aguda"      if resultado.prediction == "ENFERMEDAD AGUDA"
          else "cronica"    if resultado.prediction == "ENFERMEDAD CRÓNICA"
          else "terminal" %}

  <span class="badge {{ cls }}">{{ resultado.prediction }}</span>
  <div class="score">Clase {{ resultado.prediction_class }} / 4</div>

  <div class="meta">
    <span>🆔 <b>ID:</b> {{ resultado.prediction_id[:8] }}…</span>
    <span>🕐 <b>Hora:</b> {{ resultado.timestamp[:19].replace("T"," ") }} UTC</span><br/>
    <span>🏷 <b>Versión:</b> {{ resultado.model_version }}</span>
    <span>🌐 <b>Entorno:</b> {{ resultado.environment }}</span>
  </div>

  <div class="params">
    <strong>Parámetros evaluados:</strong><br/>
    Edad: {{ resultado.input.edad }} años ·
    Síntomas: {{ resultado.input.num_sintomas }} ·
    Días: {{ resultado.input.dias_sintomas }} ·
    Dolor: {{ resultado.input.nivel_dolor }}/10 ·
    Fiebre: {{ "Sí" if resultado.input.tiene_fiebre else "No" }} ·
    Enf. base: {{ "Sí" if resultado.input.enfermedad_base else "No" }}
  </div>
</div>
{% endif %}

<footer>Universidad Icesi — Maestría en Inteligencia Artificial Aplicada — 2026</footer>
</body>
</html>
"""


def _render(resultado=None, edad="", num_sintomas="", dias_sintomas="",
            nivel_dolor="", tiene_fiebre=False, enfermedad_base=False):
    """Renderiza el template HTML con los datos proporcionados."""
    return HTMLResponse(Template(PAGINA_HTML).render(
        environment=ENVIRONMENT,
        model_version=MODEL_VERSION,
        env_badge=f"{ENVIRONMENT.upper()} · {MODEL_VERSION}",
        edad=edad,
        num_sintomas=num_sintomas,
        dias_sintomas=dias_sintomas,
        nivel_dolor=nivel_dolor,
        tiene_fiebre=tiene_fiebre,
        enfermedad_base=enfermedad_base,
        resultado=resultado,
    ))


# ─────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────

def _ejecutar_prediccion(input_dict: dict) -> PredictionResponse:
    """
    Ejecuta la inferencia ONNX, construye la respuesta y registra en GCS.
    Reutilizado por el endpoint JSON y el formulario web.
    """
    result = predict(input_dict)

    response = PredictionResponse(
        environment=ENVIRONMENT,
        prediction=result["prediction_label"],
        prediction_class=result["prediction_class"],
        prediction_id=str(uuid.uuid4()),
        model_version=MODEL_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
        input=input_dict,
    )

    try:
        get_logger().log_prediction(response.model_dump())
    except Exception as e:
        logger.warning(f"No se pudo registrar predicción: {e}")

    return response


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home():
    """Formulario web vacío para el médico."""
    return _render()


@app.post("/predecir", response_class=HTMLResponse)
async def predecir_formulario(
    edad:            int = Form(...),
    num_sintomas:    int = Form(...),
    dias_sintomas:   int = Form(...),
    nivel_dolor:     int = Form(...),
    tiene_fiebre:    str = Form(default=None),
    enfermedad_base: str = Form(default=None),
):
    """
    Recibe los datos desde el formulario web y devuelve la página con el
    resultado de la predicción ONNX renderizado en HTML.
    La predicción también queda registrada en GCS.
    """
    fiebre = tiene_fiebre is not None
    base   = enfermedad_base is not None

    input_dict = {
        "num_sintomas":    num_sintomas,
        "dias_sintomas":   dias_sintomas,
        "nivel_dolor":     nivel_dolor,
        "tiene_fiebre":    int(fiebre),
        "enfermedad_base": int(base),
        "edad":            edad,
    }

    try:
        response = _ejecutar_prediccion(input_dict)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    except Exception as e:
        logger.error(f"Error en predicción web: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

    return _render(
        resultado=response,
        edad=edad,
        num_sintomas=num_sintomas,
        dias_sintomas=dias_sintomas,
        nivel_dolor=nivel_dolor,
        tiene_fiebre=fiebre,
        enfermedad_base=base,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(patient: PatientInput):
    """
    Predice el estado de enfermedad vía API REST (JSON).
    Cada predicción se registra en GCS (o archivo local como fallback).
    """
    try:
        return _ejecutar_prediccion(patient.model_dump())
    except FileNotFoundError as e:
        logger.error(f"Modelo no disponible: {e}")
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    except Exception as e:
        logger.error(f"Error en predicción API: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


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

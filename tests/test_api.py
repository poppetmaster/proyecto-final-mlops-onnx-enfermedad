"""
Tests de la API FastAPI.

Universidad Icesi — Proyecto Final MLOps
"""

import os
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """TestClient de FastAPI — requiere modelo disponible."""
    model_path = os.environ.get("MODEL_PATH", "models/modelo_enfermedad.onnx")
    if not os.path.exists(model_path):
        pytest.skip(f"Modelo no disponible en {model_path}")

    # Configurar entorno como local para evitar GCS
    os.environ["ENVIRONMENT"] = "local"

    from app.main import app
    with TestClient(app) as c:
        yield c


class TestAPI:
    """Pruebas de los endpoints HTTP."""

    def test_health_check(self, client):
        """GET /health debe responder 200 con status ok."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True

    def test_home_endpoint(self, client):
        """GET / debe retornar HTML con información del servicio."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert "MLOps" in resp.text

    def test_model_info_endpoint(self, client):
        """GET /model-info debe retornar metadata del modelo."""
        resp = client.get("/model-info")
        assert resp.status_code == 200
        data = resp.json()
        assert "model_version" in data
        assert "input_features" in data
        assert "output_classes" in data
        assert len(data["output_classes"]) == 5

    def test_predict_valid_input(self, client):
        """POST /predict con entrada válida debe retornar predicción."""
        payload = {
            "num_sintomas": 5,
            "dias_sintomas": 7,
            "nivel_dolor": 4,
            "tiene_fiebre": 0,
            "enfermedad_base": 0,
            "edad": 35,
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200

        data = resp.json()
        # Verificar todos los campos obligatorios
        assert "environment" in data
        assert "prediction" in data
        assert "prediction_class" in data
        assert "prediction_id" in data
        assert "model_version" in data
        assert "timestamp" in data
        assert "input" in data

        # Validar tipos y rangos
        assert data["prediction_class"] in range(5)
        assert data["prediction"] in [
            "NO ENFERMO", "ENFERMEDAD LEVE", "ENFERMEDAD AGUDA",
            "ENFERMEDAD CRÓNICA", "ENFERMEDAD TERMINAL",
        ]
        assert len(data["prediction_id"]) > 0

    def test_predict_paciente_critico(self, client):
        """Paciente con todos los factores extremos debe ser clasificado en clases altas."""
        payload = {
            "num_sintomas": 20,
            "dias_sintomas": 300,
            "nivel_dolor": 10,
            "tiene_fiebre": 1,
            "enfermedad_base": 1,
            "edad": 85,
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        assert resp.json()["prediction_class"] in [3, 4]

    def test_predict_validates_input(self, client):
        """POST /predict debe rechazar entradas fuera de rango."""
        payload = {
            "num_sintomas": -5,   # inválido
            "dias_sintomas": 7,
            "nivel_dolor": 4,
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422  # Validation error

    def test_predict_required_fields(self, client):
        """POST /predict debe requerir los campos obligatorios."""
        payload = {"num_sintomas": 5}  # faltan campos
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

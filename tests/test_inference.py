"""
Tests del módulo de inferencia con ONNX.

Universidad Icesi — Proyecto Final MLOps
"""

import os
import pytest
import numpy as np

from app.inference import predict, get_model, reset_model, CLASS_LABELS, FEATURE_ORDER


# Las clases esperadas (etiquetas textuales)
CLASES_VALIDAS = set(CLASS_LABELS.values())


@pytest.fixture(scope="module", autouse=True)
def setup_model():
    """Verifica que el modelo esté disponible antes de ejecutar tests."""
    model_path = os.environ.get("MODEL_PATH", "models/modelo_enfermedad.onnx")
    if not os.path.exists(model_path):
        pytest.skip(
            f"Modelo no disponible en {model_path}. "
            f"Ejecuta scripts/train_export_onnx.py primero "
            f"o descarga desde GCS con scripts/download_model.py"
        )
    reset_model()
    yield
    reset_model()


class TestInferencia:
    """Pruebas básicas del modelo de inferencia."""

    def test_modelo_carga_correctamente(self):
        """El modelo ONNX debe cargarse sin errores."""
        model = get_model()
        assert model is not None
        assert model.session is not None

    def test_prediccion_con_paciente_sano(self):
        """Un paciente sin síntomas debe predecirse cercano a NO ENFERMO."""
        entrada = {
            "num_sintomas": 0,
            "dias_sintomas": 0,
            "nivel_dolor": 0,
            "tiene_fiebre": 0,
            "enfermedad_base": 0,
            "edad": 25,
        }
        resultado = predict(entrada)
        assert "prediction_class" in resultado
        assert "prediction_label" in resultado
        assert resultado["prediction_class"] == 0  # NO ENFERMO
        assert resultado["prediction_label"] == "NO ENFERMO"

    def test_prediccion_paciente_critico(self):
        """Paciente con todos los factores de riesgo debe predecirse en clases altas."""
        entrada = {
            "num_sintomas": 20,
            "dias_sintomas": 200,
            "nivel_dolor": 10,
            "tiene_fiebre": 1,
            "enfermedad_base": 1,
            "edad": 85,
        }
        resultado = predict(entrada)
        # Debe ser una de las clases más graves (CRÓNICA o TERMINAL)
        assert resultado["prediction_class"] in [3, 4]

    def test_prediccion_retorna_clase_valida(self):
        """La predicción siempre debe estar entre las 5 clases definidas."""
        entrada = {
            "num_sintomas": 5,
            "dias_sintomas": 7,
            "nivel_dolor": 4,
            "tiene_fiebre": 0,
            "enfermedad_base": 0,
            "edad": 40,
        }
        resultado = predict(entrada)
        assert resultado["prediction_class"] in range(5)
        assert resultado["prediction_label"] in CLASES_VALIDAS

    def test_prediccion_con_diferentes_entradas(self):
        """Múltiples predicciones consecutivas deben funcionar."""
        entradas = [
            {"num_sintomas": 1, "dias_sintomas": 1, "nivel_dolor": 1,
             "tiene_fiebre": 0, "enfermedad_base": 0, "edad": 30},
            {"num_sintomas": 10, "dias_sintomas": 30, "nivel_dolor": 7,
             "tiene_fiebre": 1, "enfermedad_base": 1, "edad": 60},
            {"num_sintomas": 5, "dias_sintomas": 5, "nivel_dolor": 4,
             "tiene_fiebre": 0, "enfermedad_base": 0, "edad": 40},
        ]
        for entrada in entradas:
            resultado = predict(entrada)
            assert resultado["prediction_class"] in range(5)
            assert resultado["prediction_label"] in CLASES_VALIDAS

    def test_feature_order_definido(self):
        """El orden de features debe estar definido y ser de 6 elementos."""
        assert len(FEATURE_ORDER) == 6
        assert "num_sintomas" in FEATURE_ORDER
        assert "edad" in FEATURE_ORDER

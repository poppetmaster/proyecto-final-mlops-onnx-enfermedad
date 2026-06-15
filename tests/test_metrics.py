"""
Tests de métricas del modelo.

Valida que el modelo ONNX mantenga un nivel de desempeño aceptable
sobre el conjunto de prueba descargado desde GCS.

Universidad Icesi — Proyecto Final MLOps
"""

import os
import pytest
import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.metrics import accuracy_score, f1_score


FEATURE_ORDER = [
    "num_sintomas",
    "dias_sintomas",
    "nivel_dolor",
    "tiene_fiebre",
    "enfermedad_base",
    "edad",
]


@pytest.fixture(scope="module")
def test_data():
    """Carga el CSV de prueba."""
    path = os.environ.get("TEST_DATA_PATH", "data/test_data.csv")
    if not os.path.exists(path):
        pytest.skip(
            f"Test data no disponible en {path}. "
            f"Ejecuta scripts/train_export_onnx.py o "
            f"descarga desde GCS con scripts/download_test_data.py"
        )
    df = pd.read_csv(path)
    assert "y_true" in df.columns, "El CSV debe tener columna 'y_true'"
    return df


@pytest.fixture(scope="module")
def onnx_session():
    """Carga el modelo ONNX."""
    path = os.environ.get("MODEL_PATH", "models/modelo_enfermedad.onnx")
    if not os.path.exists(path):
        pytest.skip(f"Modelo no disponible en {path}")
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


@pytest.fixture(scope="module")
def predictions(onnx_session, test_data):
    """Calcula predicciones sobre todo el set de prueba."""
    X = test_data[FEATURE_ORDER].values.astype(np.float32)
    input_name = onnx_session.get_inputs()[0].name
    outputs = onnx_session.run(None, {input_name: X})
    return outputs[0].astype(int).flatten()


class TestMetricas:
    """Validación del desempeño del modelo."""

    THRESHOLD = float(os.environ.get("METRIC_THRESHOLD", "0.80"))

    def test_accuracy_supera_umbral(self, test_data, predictions):
        """La accuracy del modelo debe ser >= 0.80."""
        y_true = test_data["y_true"].values
        acc = accuracy_score(y_true, predictions)
        print(f"\n  Accuracy obtenida: {acc:.4f} (umbral: {self.THRESHOLD})")
        assert acc >= self.THRESHOLD, (
            f"Accuracy {acc:.4f} por debajo del umbral {self.THRESHOLD}"
        )

    def test_f1_weighted_supera_umbral(self, test_data, predictions):
        """El F1 weighted del modelo debe ser >= 0.80."""
        y_true = test_data["y_true"].values
        f1 = f1_score(y_true, predictions, average="weighted", zero_division=0)
        print(f"\n  F1 (weighted) obtenido: {f1:.4f} (umbral: {self.THRESHOLD})")
        assert f1 >= self.THRESHOLD, (
            f"F1 weighted {f1:.4f} por debajo del umbral {self.THRESHOLD}"
        )

    def test_predicciones_dentro_de_rango(self, predictions):
        """Todas las predicciones deben estar en el rango 0-4."""
        assert predictions.min() >= 0
        assert predictions.max() <= 4

    def test_modelo_predice_multiples_clases(self, predictions):
        """El modelo debe predecir al menos 3 clases distintas (no degenerar)."""
        unique = np.unique(predictions)
        assert len(unique) >= 3, (
            f"El modelo solo predice {len(unique)} clase(s): {unique}"
        )

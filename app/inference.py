"""
Módulo de inferencia con ONNX Runtime.

Carga el modelo ONNX desde la ruta indicada por MODEL_PATH y expone
la función predict() que recibe los datos del paciente y retorna la
clase predicha junto con su etiqueta textual.
"""

import os
import logging
import numpy as np
import onnxruntime as ort
from typing import Optional

logger = logging.getLogger(__name__)


# Mapeo de clases numéricas a etiquetas textuales
CLASS_LABELS = {
    0: "NO ENFERMO",
    1: "ENFERMEDAD LEVE",
    2: "ENFERMEDAD AGUDA",
    3: "ENFERMEDAD CRÓNICA",
    4: "ENFERMEDAD TERMINAL",
}

# Orden de las features que espera el modelo ONNX
FEATURE_ORDER = [
    "num_sintomas",
    "dias_sintomas",
    "nivel_dolor",
    "tiene_fiebre",
    "enfermedad_base",
    "edad",
]


class ONNXModel:
    """Wrapper para el modelo ONNX cargado en memoria."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self._load()

    def _load(self) -> None:
        """Carga el modelo ONNX desde disco."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"No se encontró el modelo ONNX en: {self.model_path}. "
                f"Asegúrate de descargarlo desde GCS antes de iniciar la app."
            )
        logger.info(f"Cargando modelo ONNX desde {self.model_path}")
        self.session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        logger.info(f"Modelo cargado. Input name: {self.input_name}")

    def predict_class(self, features: np.ndarray) -> int:
        """Ejecuta inferencia y retorna la clase predicha (0-4)."""
        if self.session is None:
            raise RuntimeError("Modelo no cargado")

        # Asegurar shape (1, n_features) y dtype float32
        if features.ndim == 1:
            features = features.reshape(1, -1)
        features = features.astype(np.float32)

        outputs = self.session.run(None, {self.input_name: features})
        # El primer output es la clase predicha
        pred = outputs[0]
        return int(pred[0])


# Instancia global del modelo (lazy initialization)
_model_instance: Optional[ONNXModel] = None


def get_model() -> ONNXModel:
    """Retorna la instancia singleton del modelo, cargándolo si es necesario."""
    global _model_instance
    if _model_instance is None:
        model_path = os.environ.get("MODEL_PATH", "models/modelo_enfermedad.onnx")
        _model_instance = ONNXModel(model_path)
    return _model_instance


def reset_model() -> None:
    """Resetea el modelo (útil para tests)."""
    global _model_instance
    _model_instance = None


def predict(input_data: dict) -> dict:
    """
    Ejecuta predicción a partir de un diccionario con los datos del paciente.

    Parámetros
    ----------
    input_data : dict
        Diccionario con las features del paciente (ver FEATURE_ORDER).

    Retorna
    -------
    dict con:
        - prediction_class : int  → 0-4
        - prediction_label : str  → Etiqueta textual
    """
    # Construir vector de features en el orden esperado
    features = np.array([
        [input_data[feature] for feature in FEATURE_ORDER]
    ], dtype=np.float32)

    model = get_model()
    pred_class = model.predict_class(features)
    pred_label = CLASS_LABELS.get(pred_class, "DESCONOCIDO")

    return {
        "prediction_class": pred_class,
        "prediction_label": pred_label,
    }

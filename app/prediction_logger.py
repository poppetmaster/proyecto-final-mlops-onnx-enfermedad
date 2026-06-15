"""
Logger de predicciones.

Agrega cada predicción como una línea en un archivo TXT en Google Cloud
Storage. Si no hay credenciales o GCS no está disponible, hace fallback
a un archivo local para que la aplicación nunca falle por logging.

Variables de entorno:
    - ENVIRONMENT       : dev | prod | local
    - GCS_LOGS_BUCKET   : nombre del bucket de logs
    - PREDICTIONS_FILE  : nombre del archivo (predicciones_dev.txt, etc.)
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Importación condicional de google-cloud-storage
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    logger.warning("google-cloud-storage no instalado; logger en modo local únicamente")


class PredictionLogger:
    """Registra predicciones en GCS o en archivo local como fallback."""

    def __init__(self):
        self.environment = os.environ.get("ENVIRONMENT", "local")
        self.bucket_name = os.environ.get("GCS_LOGS_BUCKET", "")
        self.predictions_file = os.environ.get(
            "PREDICTIONS_FILE",
            f"predicciones_{self.environment}.txt",
        )
        self.local_fallback_path = os.environ.get(
            "LOCAL_LOG_PATH",
            f"/tmp/{self.predictions_file}",
        )

        self._gcs_client = None
        self._use_gcs = self._can_use_gcs()

        if self._use_gcs:
            logger.info(
                f"Logger en modo GCS: gs://{self.bucket_name}/{self.predictions_file}"
            )
        else:
            logger.info(f"Logger en modo LOCAL: {self.local_fallback_path}")

    def _can_use_gcs(self) -> bool:
        """Verifica si se puede usar GCS."""
        if not GCS_AVAILABLE:
            return False
        if not self.bucket_name:
            return False
        try:
            self._gcs_client = storage.Client()
            # Validación liviana — no hace request hasta que se use el bucket
            return True
        except Exception as e:
            logger.warning(f"No se pudo inicializar GCS client: {e}")
            return False

    def log_prediction(self, prediction_data: dict) -> bool:
        """
        Registra una predicción.

        Retorna True si se logró registrar (GCS o local), False si falló todo.
        """
        line = self._format_line(prediction_data)

        if self._use_gcs:
            try:
                self._append_to_gcs(line)
                return True
            except Exception as e:
                logger.error(f"Error al escribir en GCS, fallback a local: {e}")
                # Cae al fallback local

        try:
            self._append_to_local(line)
            return True
        except Exception as e:
            logger.error(f"Error al escribir en archivo local: {e}")
            return False

    def _format_line(self, data: dict) -> str:
        """Formatea la predicción como una línea JSON con timestamp."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **data,
        }
        return json.dumps(record, ensure_ascii=False) + "\n"

    def _append_to_gcs(self, line: str) -> None:
        """
        Agrega una línea al blob en GCS.

        GCS no soporta append nativo, así que se descarga el contenido,
        se concatena la nueva línea y se sube de vuelta.
        Para alto volumen, se recomendaría usar Pub/Sub + BigQuery.
        """
        bucket = self._gcs_client.bucket(self.bucket_name)
        blob = bucket.blob(self.predictions_file)

        current = ""
        if blob.exists():
            current = blob.download_as_text(encoding="utf-8")

        blob.upload_from_string(
            current + line,
            content_type="text/plain; charset=utf-8",
        )

    def _append_to_local(self, line: str) -> None:
        """Agrega una línea al archivo local."""
        os.makedirs(os.path.dirname(self.local_fallback_path) or ".", exist_ok=True)
        with open(self.local_fallback_path, "a", encoding="utf-8") as f:
            f.write(line)


# Singleton del logger
_logger_instance: Optional[PredictionLogger] = None


def get_logger() -> PredictionLogger:
    """Retorna la instancia singleton del logger."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = PredictionLogger()
    return _logger_instance


def reset_logger() -> None:
    """Resetea el logger (útil para tests)."""
    global _logger_instance
    _logger_instance = None

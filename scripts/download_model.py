"""
Descarga el modelo ONNX desde Google Cloud Storage.

Variables de entorno requeridas:
    - GCS_MODEL_BUCKET : nombre del bucket
    - GCS_MODEL_BLOB   : ruta del blob dentro del bucket (ej: modelo_enfermedad.onnx)
    - MODEL_PATH       : ruta local destino (ej: models/modelo_enfermedad.onnx)

Uso:
    python scripts/download_model.py
"""

import os
import sys
from google.cloud import storage


def main():
    bucket_name = os.environ.get("GCS_MODEL_BUCKET")
    blob_name = os.environ.get("GCS_MODEL_BLOB", "modelo_enfermedad.onnx")
    dest_path = os.environ.get("MODEL_PATH", "models/modelo_enfermedad.onnx")

    if not bucket_name:
        print("❌ ERROR: GCS_MODEL_BUCKET no está definido")
        sys.exit(1)

    print(f"Descargando gs://{bucket_name}/{blob_name} → {dest_path}")

    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        if not blob.exists():
            print(f"❌ ERROR: El blob no existe en GCS: gs://{bucket_name}/{blob_name}")
            sys.exit(1)

        blob.download_to_filename(dest_path)
        size_kb = os.path.getsize(dest_path) / 1024
        print(f"✅ Modelo descargado correctamente ({size_kb:.1f} KB)")

    except Exception as e:
        print(f"❌ ERROR al descargar modelo: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

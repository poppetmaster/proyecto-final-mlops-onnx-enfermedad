"""
Descarga el archivo de datos de prueba desde Google Cloud Storage.

Variables de entorno requeridas:
    - GCS_DATA_BUCKET    : nombre del bucket
    - GCS_TEST_DATA_BLOB : ruta del blob (ej: test_data.csv)
    - TEST_DATA_PATH     : ruta local destino (ej: data/test_data.csv)

Uso:
    python scripts/download_test_data.py
"""

import os
import sys
from google.cloud import storage


def main():
    bucket_name = os.environ.get("GCS_DATA_BUCKET")
    blob_name = os.environ.get("GCS_TEST_DATA_BLOB", "test_data.csv")
    dest_path = os.environ.get("TEST_DATA_PATH", "data/test_data.csv")

    if not bucket_name:
        print("❌ ERROR: GCS_DATA_BUCKET no está definido")
        sys.exit(1)

    print(f"Descargando gs://{bucket_name}/{blob_name} → {dest_path}")

    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        if not blob.exists():
            print(f"❌ ERROR: El blob no existe: gs://{bucket_name}/{blob_name}")
            sys.exit(1)

        blob.download_to_filename(dest_path)
        size_kb = os.path.getsize(dest_path) / 1024
        print(f"✅ Test data descargado correctamente ({size_kb:.1f} KB)")

    except Exception as e:
        print(f"❌ ERROR al descargar test data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

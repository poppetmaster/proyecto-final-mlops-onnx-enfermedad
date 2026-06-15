"""
Validación del modelo ONNX contra el conjunto de prueba.

Carga el modelo desde MODEL_PATH y el CSV desde TEST_DATA_PATH,
ejecuta inferencia y verifica que la métrica (F1 weighted por defecto)
supere el umbral mínimo configurado en METRIC_THRESHOLD.

Sale con código 0 si pasa, 1 si falla → útil para CI/CD.

Variables de entorno:
    - MODEL_PATH       : ruta al modelo ONNX
    - TEST_DATA_PATH   : ruta al CSV de prueba
    - METRIC_THRESHOLD : umbral mínimo (default 0.80)
    - METRIC_NAME      : 'accuracy' o 'f1_weighted' (default 'f1_weighted')

Uso:
    python scripts/validate_model.py
"""

import os
import sys
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


def main():
    model_path = os.environ.get("MODEL_PATH", "models/modelo_enfermedad.onnx")
    data_path = os.environ.get("TEST_DATA_PATH", "data/test_data.csv")
    threshold = float(os.environ.get("METRIC_THRESHOLD", "0.80"))
    metric_name = os.environ.get("METRIC_NAME", "f1_weighted")

    print("=" * 60)
    print(" Validación del modelo ONNX")
    print("=" * 60)
    print(f" Modelo     : {model_path}")
    print(f" Test data  : {data_path}")
    print(f" Métrica    : {metric_name}")
    print(f" Umbral     : {threshold}")
    print("=" * 60)

    # Cargar modelo
    if not os.path.exists(model_path):
        print(f"❌ Modelo no encontrado: {model_path}")
        sys.exit(1)
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Cargar datos
    if not os.path.exists(data_path):
        print(f"❌ Test data no encontrada: {data_path}")
        sys.exit(1)
    df = pd.read_csv(data_path)
    if "y_true" not in df.columns:
        print(f"❌ El CSV debe tener columna 'y_true'")
        sys.exit(1)

    X = df[FEATURE_ORDER].values.astype(np.float32)
    y_true = df["y_true"].values

    print(f"\n Muestras de prueba: {len(df)}")

    # Inferencia
    outputs = session.run(None, {input_name: X})
    y_pred = outputs[0].astype(int).flatten()

    # Métricas
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n Accuracy       : {acc:.4f}")
    print(f" F1 (weighted)  : {f1:.4f}")

    metric_value = f1 if metric_name == "f1_weighted" else acc

    print(f"\n Métrica seleccionada ({metric_name}): {metric_value:.4f}")
    print(f" Umbral mínimo                       : {threshold:.4f}")

    if metric_value >= threshold:
        print(f"\n ✅ APROBADO: La métrica supera el umbral")
        sys.exit(0)
    else:
        print(f"\n ❌ FALLÓ: La métrica está por debajo del umbral")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Script de entrenamiento y exportación a ONNX.

Genera un dataset sintético de pacientes, entrena un RandomForestClassifier
y exporta el modelo a formato ONNX, junto con un CSV de prueba.

Outputs:
    - artifacts/modelo_enfermedad.onnx
    - artifacts/test_data.csv

Uso:
    python scripts/train_export_onnx.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# ─────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────

OUTPUT_DIR = "artifacts"
MODEL_FILE = os.path.join(OUTPUT_DIR, "modelo_enfermedad.onnx")
TEST_DATA_FILE = os.path.join(OUTPUT_DIR, "test_data.csv")

N_SAMPLES = 5000          # Tamaño del dataset sintético
TEST_SIZE = 0.2           # Proporción para test
RANDOM_STATE = 42

FEATURE_NAMES = [
    "num_sintomas",
    "dias_sintomas",
    "nivel_dolor",
    "tiene_fiebre",
    "enfermedad_base",
    "edad",
]

CLASS_NAMES = {
    0: "NO ENFERMO",
    1: "ENFERMEDAD LEVE",
    2: "ENFERMEDAD AGUDA",
    3: "ENFERMEDAD CRÓNICA",
    4: "ENFERMEDAD TERMINAL",
}


# ─────────────────────────────────────────────
# Generación del dataset sintético
# ─────────────────────────────────────────────

def calcular_puntaje(num_sintomas, dias_sintomas, nivel_dolor,
                    tiene_fiebre, enfermedad_base, edad):
    """Sistema de puntaje basado en reglas clínicas (referencia)."""
    p = 0.0
    p += min(num_sintomas / 10, 1.0) * 25

    if dias_sintomas <= 2:
        p += 4
    elif dias_sintomas <= 7:
        p += 10
    elif dias_sintomas <= 30:
        p += 16
    else:
        p += 20

    p += (nivel_dolor / 10) * 25

    if tiene_fiebre:
        p += 10
    if enfermedad_base:
        p += 10

    if edad < 5 or edad > 65:
        p += 10
    elif edad < 12 or edad > 50:
        p += 5

    return max(0.0, min(p, 100.0))


def puntaje_a_clase(p):
    """Mapea puntaje a clase numérica 0-4."""
    if p < 20:   return 0
    if p < 45:   return 1
    if p < 65:   return 2
    if p < 85:   return 3
    return 4


def generar_dataset(n=N_SAMPLES, seed=RANDOM_STATE):
    """
    Genera un dataset sintético balanceado.

    Estrategia: en lugar de muestrear features uniformemente (lo cual
    produce sesgo hacia clases medias-altas por construcción del puntaje),
    se muestrea condicionando aproximadamente a cada clase, así obtenemos
    una distribución más realista para entrenamiento.
    """
    rng = np.random.default_rng(seed)

    # Distribución objetivo (más realista en medicina general):
    # NO ENFERMO 30%, LEVE 30%, AGUDA 20%, CRÓNICA 15%, TERMINAL 5%
    target_dist = [0.30, 0.30, 0.20, 0.15, 0.05]
    n_per_class = [int(n * p) for p in target_dist]
    n_per_class[0] += n - sum(n_per_class)  # ajuste por redondeo

    # Rangos típicos por clase (sin ser absolutos — se permite solapamiento)
    rangos_por_clase = {
        0: {"sint": (0, 3), "dias": (0, 5), "dolor": (0, 3),
            "fiebre": (0, 0.1), "base": (0, 0.05), "edad": (15, 50)},
        1: {"sint": (2, 6), "dias": (3, 12), "dolor": (2, 5),
            "fiebre": (0, 0.3), "base": (0, 0.2), "edad": (15, 55)},
        2: {"sint": (5, 12), "dias": (5, 25), "dolor": (4, 8),
            "fiebre": (0, 0.7), "base": (0, 0.3), "edad": (20, 65)},
        3: {"sint": (8, 16), "dias": (30, 120), "dolor": (5, 9),
            "fiebre": (0, 0.5), "base": (0, 0.7), "edad": (35, 75)},
        4: {"sint": (12, 20), "dias": (60, 365), "dolor": (8, 10),
            "fiebre": (0, 0.8), "base": (0, 0.95), "edad": (50, 95)},
    }

    rows = []
    labels = []
    for cls, count in enumerate(n_per_class):
        r = rangos_por_clase[cls]
        for _ in range(count):
            sint = rng.integers(r["sint"][0], r["sint"][1] + 1)
            dias = rng.integers(r["dias"][0], r["dias"][1] + 1)
            dolor = rng.integers(r["dolor"][0], r["dolor"][1] + 1)
            fiebre = int(rng.random() < r["fiebre"][1])
            base = int(rng.random() < r["base"][1])
            edad = rng.integers(r["edad"][0], r["edad"][1] + 1)

            # Calcular puntaje resultante (con ruido)
            p = calcular_puntaje(sint, dias, dolor, fiebre, base, edad)
            p_ruidoso = p + rng.normal(0, 4)
            cls_resultante = puntaje_a_clase(p_ruidoso)

            rows.append([sint, dias, dolor, fiebre, base, edad])
            labels.append(cls_resultante)  # usar la clase derivada (con ruido)

    # Mezclar
    indices = rng.permutation(len(rows))
    rows = [rows[i] for i in indices]
    labels = [labels[i] for i in indices]

    X = pd.DataFrame(rows, columns=[
        "num_sintomas", "dias_sintomas", "nivel_dolor",
        "tiene_fiebre", "enfermedad_base", "edad",
    ])
    y = np.array(labels)

    return X, y


# ─────────────────────────────────────────────
# Entrenamiento y exportación
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" Entrenamiento y exportación ONNX")
    print(" Universidad Icesi — Proyecto Final MLOps")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Generar dataset
    print(f"\n[1/4] Generando dataset sintético ({N_SAMPLES} muestras)...")
    X, y = generar_dataset()
    print(f"      Distribución de clases:")
    for c, name in CLASS_NAMES.items():
        count = int(np.sum(y == c))
        print(f"        Clase {c} ({name:22s}): {count:4d}  ({100*count/len(y):.1f}%)")

    # 2. Split train/test
    print(f"\n[2/4] Dividiendo train/test ({1-TEST_SIZE:.0%}/{TEST_SIZE:.0%})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    # 3. Entrenar
    print(f"\n[3/4] Entrenando RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X_train.values, y_train)

    y_pred = model.predict(X_test.values)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"      Accuracy: {acc:.4f}")
    print(f"      F1 (weighted): {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=list(CLASS_NAMES.values()), zero_division=0)}")

    # 4. Exportar a ONNX
    print(f"\n[4/4] Exportando modelo a ONNX...")
    initial_type = [("float_input", FloatTensorType([None, len(FEATURE_NAMES)]))]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=15,
        options={id(model): {"zipmap": False}},  # Salida más simple
    )

    with open(MODEL_FILE, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"      ✅ Modelo guardado en: {MODEL_FILE}")
    print(f"      Tamaño: {os.path.getsize(MODEL_FILE) / 1024:.1f} KB")

    # Guardar test data
    test_df = X_test.copy()
    test_df["y_true"] = y_test
    test_df.to_csv(TEST_DATA_FILE, index=False)
    print(f"      ✅ Test data guardado en: {TEST_DATA_FILE}")
    print(f"      Muestras de prueba: {len(test_df)}")

    print("\n" + "=" * 60)
    print(" ✅ Entrenamiento completado")
    print("=" * 60)
    print(f"\nPróximos pasos:")
    print(f"  1. Subir {MODEL_FILE} al bucket GCS_MODEL_BUCKET")
    print(f"  2. Subir {TEST_DATA_FILE} al bucket GCS_DATA_BUCKET")
    print(f"  3. Configurar variables en GitHub Secrets y Cloud Run")


if __name__ == "__main__":
    main()

# MLOps · Predicción de Estado de Enfermedad con ONNX

**Universidad Icesi — Maestría en Inteligencia Artificial Aplicada**
**Proyecto Final · MLOps**

---

## 1. Descripción del proyecto

Pipeline MLOps end-to-end para una API de predicción de estado de enfermedad
basada en datos clínicos. El modelo es un `RandomForestClassifier` exportado a
**ONNX** que se sirve a través de **FastAPI** sobre **Cloud Run**, con dos
ambientes (`dev`/`prod`), entrega continua mediante **GitHub Actions** y
artefactos versionados en **Google Cloud Storage**.

### Características clave

| Capacidad | Implementación |
|---|---|
| Modelo en formato portable | ONNX (onnxruntime, sin dependencia de sklearn en runtime) |
| Versionado de artefactos | Modelo y datos de prueba viven en GCS, no en el repo |
| CI/CD por rama | `dev` → Cloud Run DEV · `prod` → Cloud Run PROD |
| Validación de calidad | Métrica mínima F1 weighted ≥ 0.80 antes de promover |
| Trazabilidad | Cada predicción se registra en GCS con timestamp y UUID |
| Despliegue serverless | Cloud Run con autoscaling de 0 a N |

---

## 2. Arquitectura

```
┌──────────────────┐       ┌────────────────────┐
│   GitHub Repo    │ push  │  GitHub Actions    │
│  (dev / prod)    │──────▶│   CI/CD Workflow   │
└──────────────────┘       └─────────┬──────────┘
                                     │
                  ┌──────────────────┼──────────────────┐
                  │                  │                  │
                  ▼                  ▼                  ▼
        ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
        │  Job: TEST     │  │ Job: BUILD     │  │ Job: DEPLOY    │
        │  - pytest      │  │ - docker build │  │ - gcloud run   │
        │  - validate    │  │ - push image   │  │   deploy       │
        └───────┬────────┘  └───────┬────────┘  └───────┬────────┘
                │                   │                   │
                ▼                   ▼                   ▼
        ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
        │ GCS: modelo    │  │ Artifact       │  │ Cloud Run      │
        │ GCS: testdata  │  │ Registry       │  │ DEV / PROD     │
        └────────────────┘  └────────────────┘  └───────┬────────┘
                                                        │
                                              ┌─────────▼────────┐
                                              │  GCS: logs       │
                                              │  predicciones_*.txt │
                                              └──────────────────┘
```

---

## 3. Ramas y entornos

| Rama | Cloud Run | Buckets | Umbral métrica |
|---|---|---|---|
| `dev` | `mlops-enfermedad-dev` | `*-dev` | 0.80 |
| `prod` | `mlops-enfermedad-prod` | `*-prod` | 0.85 (recomendado) |

El workflow se dispara con `git push` a cualquiera de las dos ramas y
selecciona automáticamente el conjunto de buckets y el nombre del servicio
correspondiente.

---

## 4. Modelo ONNX

- **Algoritmo:** `RandomForestClassifier` (sklearn) → exportado con `skl2onnx`
- **Features:** `num_sintomas`, `dias_sintomas`, `nivel_dolor`, `tiene_fiebre`, `enfermedad_base`, `edad`
- **Clases:** 0=NO ENFERMO · 1=LEVE · 2=AGUDA · 3=CRÓNICA · 4=TERMINAL
- **Runtime:** `onnxruntime` con `CPUExecutionProvider`
- **Ubicación:** `gs://<GCS_MODEL_BUCKET>/modelo_enfermedad.onnx` (no en repo)

Para regenerar:

```bash
python scripts/train_export_onnx.py
# Genera: artifacts/modelo_enfermedad.onnx y artifacts/test_data.csv
```

---

## 5. Buckets de Google Cloud Storage

| Bucket | Contenido | Usado por |
|---|---|---|
| `icesi-mlops-models-{env}` | `modelo_enfermedad.onnx` | Pipeline (descarga) |
| `icesi-mlops-data-{env}` | `test_data.csv` | Pipeline (validación) |
| `icesi-mlops-logs-{env}` | `predicciones_{env}.txt` | Runtime (registro) |

Crear los buckets manualmente o con:

```bash
gcloud storage buckets create gs://icesi-mlops-models-dev --location=us-central1
gcloud storage buckets create gs://icesi-mlops-data-dev   --location=us-central1
gcloud storage buckets create gs://icesi-mlops-logs-dev   --location=us-central1
# Repetir para prod
```

Subir artefactos iniciales:

```bash
gcloud storage cp artifacts/modelo_enfermedad.onnx gs://icesi-mlops-models-dev/
gcloud storage cp artifacts/test_data.csv          gs://icesi-mlops-data-dev/
```

---

## 6. GitHub Actions — Pipeline CI/CD

El workflow `.github/workflows/ci-cd.yml` tiene dos jobs:

### Job 1 — `test`
1. Checkout
2. Setup Python 3.11
3. Instalar requirements
4. Autenticarse en GCP (secret `GCP_SERVICE_ACCOUNT_KEY`)
5. Descargar modelo desde GCS
6. Descargar test data desde GCS
7. Ejecutar `pytest tests/`
8. Ejecutar `scripts/validate_model.py` (umbral F1 ≥ 0.80)

### Job 2 — `build-promote` (depende de `test`)
1. Autenticarse en GCP
2. Configurar Docker para Artifact Registry
3. Descargar modelo desde GCS (para incluirlo en la imagen)
4. `docker build` con tag según rama (`dev`/`prod`) + SHA
5. `docker push` a Artifact Registry
6. `gcloud run deploy` al servicio correspondiente
7. Inyectar variables de entorno al servicio Cloud Run

### Secrets de GitHub requeridos

```
GCP_SERVICE_ACCOUNT_KEY     → JSON de service account (con permisos GCS, Artifact Registry, Cloud Run)
GCP_PROJECT_ID              → ID del proyecto en GCP
GCS_MODEL_BUCKET_DEV        → nombre del bucket de modelos (dev)
GCS_MODEL_BUCKET_PROD       → nombre del bucket de modelos (prod)
GCS_DATA_BUCKET_DEV         → nombre del bucket de datos (dev)
GCS_DATA_BUCKET_PROD        → nombre del bucket de datos (prod)
GCS_LOGS_BUCKET_DEV         → nombre del bucket de logs (dev)
GCS_LOGS_BUCKET_PROD        → nombre del bucket de logs (prod)
```

### Permisos del Service Account

```
roles/storage.objectAdmin
roles/artifactregistry.writer
roles/run.admin
roles/iam.serviceAccountUser
```

---

## 7. Pruebas

### Estructura

| Test | Qué valida |
|---|---|
| `tests/test_inference.py` | El modelo carga y responde con entradas definidas |
| `tests/test_api.py` | Endpoints `/health` y `/predict` funcionan correctamente |
| `tests/test_metrics.py` | F1 weighted ≥ 0.80 sobre `test_data.csv` |

### Ejecutar localmente

```bash
# Requiere artifacts/ generados con train_export_onnx.py
export MODEL_PATH=artifacts/modelo_enfermedad.onnx
export TEST_DATA_PATH=artifacts/test_data.csv
pytest tests/ -v
```

---

## 8. Métrica y umbral

| Métrica | Valor mínimo | Notas |
|---|---|---|
| **F1 weighted** | ≥ 0.80 | Robusta ante clases desbalanceadas |
| Accuracy | ≥ 0.80 | Métrica secundaria |

Si la validación falla, el pipeline **se detiene** y no se hace push de la
imagen ni deploy. Esto evita publicar un modelo degradado.

---

## 9. Endpoints

### Dev

```
https://mlops-enfermedad-dev-XXXXXX-uc.a.run.app
```

### Prod

```
https://mlops-enfermedad-prod-XXXXXX-uc.a.run.app
```

(las URLs se imprimen en el summary de cada deploy de GitHub Actions)

### Lista de endpoints

| Método | Path | Descripción |
|---|---|---|
| GET  | `/` | Página HTML de bienvenida |
| GET  | `/health` | Health check (usado por Cloud Run) |
| GET  | `/model-info` | Versión, features y clases del modelo |
| GET  | `/docs` | Swagger UI |
| POST | `/predict` | Predicción de estado de enfermedad |

---

## 10. Ejecución local

### Opción A — Sin Docker

```bash
# 1. Setup
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac
pip install -r requirements.txt

# 2. Generar modelo y test data
python scripts/train_export_onnx.py

# 3. Configurar variables
set MODEL_PATH=artifacts/modelo_enfermedad.onnx
set ENVIRONMENT=local
set MODEL_VERSION=v1.0.0-local

# 4. Levantar API
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

### Opción B — Con Docker

```bash
# 1. Generar el modelo primero (Docker lo copia durante el build)
python scripts/train_export_onnx.py
mkdir models
copy artifacts\modelo_enfermedad.onnx models\modelo_enfermedad.onnx

# 2. Build
docker build -t mlops-enfermedad:local .

# 3. Run
docker run -d -p 8080:8080 ^
  -e ENVIRONMENT=local ^
  -e MODEL_VERSION=v1.0.0-local ^
  --name mlops mlops-enfermedad:local
```

Visitar: http://localhost:8080

---

## 11. Probar con curl

### Health check

```bash
curl http://localhost:8080/health
```

### Predicción (PowerShell)

```powershell
curl -X POST http://localhost:8080/predict `
     -H "Content-Type: application/json" `
     -d '{\"num_sintomas\":8,\"dias_sintomas\":15,\"nivel_dolor\":7,\"tiene_fiebre\":1,\"enfermedad_base\":0,\"edad\":50}'
```

### Predicción (CMD)

```cmd
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d "{\"num_sintomas\":8,\"dias_sintomas\":15,\"nivel_dolor\":7,\"tiene_fiebre\":1,\"enfermedad_base\":0,\"edad\":50}"
```

### Respuesta esperada

```json
{
  "environment": "dev",
  "prediction": "ENFERMEDAD AGUDA",
  "prediction_class": 2,
  "prediction_id": "a1b2c3d4-...",
  "model_version": "v1.0.0-dev",
  "timestamp": "2026-06-14T16:30:00+00:00",
  "input": { ... }
}
```

---

## 12. Validar logs en GCS

Cada predicción se registra en `gs://<GCS_LOGS_BUCKET>/predicciones_{env}.txt`
como una línea JSON.

### Ver contenido

```bash
gcloud storage cat gs://icesi-mlops-logs-dev/predicciones_dev.txt
```

### Descargar localmente

```bash
gcloud storage cp gs://icesi-mlops-logs-dev/predicciones_dev.txt ./
```

### Contar predicciones del día

```bash
gcloud storage cat gs://icesi-mlops-logs-dev/predicciones_dev.txt | grep "2026-06-14" | wc -l
```

---

## 13. Limitaciones

- **Append en GCS**: GCS no soporta append nativo. El logger descarga el blob,
  agrega la línea y lo sube de nuevo. Para alto volumen se recomendaría
  Pub/Sub + BigQuery o Cloud Logging.
- **Cold starts**: Cloud Run con `min-instances=0` puede tener latencia inicial
  ~3-5s. Si se requiere baja latencia constante, configurar `min-instances=1`.
- **Sin reentrenamiento automático**: el proyecto cubre el pipeline desde
  modelo entrenado hasta despliegue, pero no incluye reentrenamiento periódico
  ni detección de drift en producción.
- **Modelo sintético**: los datos son generados a partir de reglas conocidas,
  por lo que las métricas son artificialmente altas. Con datos reales habría
  que ajustar el umbral.
- **Autenticación abierta**: Cloud Run está con `--allow-unauthenticated`
  para facilitar la demo. En un entorno real se debería usar IAM o IAP.

---

## 14. Plan de demo (10 minutos)

Ver `docs/demo_plan.md` para el guion detallado.

**Resumen del flujo de demo:**

1. **(1 min)** Mostrar repositorio y ramas `dev`/`prod`
2. **(1 min)** Mostrar el GitHub Actions workflow funcionando
3. **(1 min)** Mostrar los buckets en GCS con modelo, datos y logs
4. **(2 min)** Mostrar la imagen en Artifact Registry y los servicios en Cloud Run
5. **(2 min)** Probar el endpoint DEV con curl y mostrar la respuesta
6. **(1 min)** Probar el endpoint PROD con curl
7. **(1 min)** Mostrar el archivo `predicciones_dev.txt` actualizado en GCS
8. **(1 min)** Hacer un cambio en `dev`, hacer push, ver el pipeline rodar

---

## 15. Estructura del repositorio

```
proyecto-final-mlops-onnx-enfermedad/
├── README.md                    # Este archivo
├── CHANGELOG.md                 # Historial de cambios
├── Dockerfile                   # Imagen compatible con Cloud Run
├── requirements.txt
├── .gitignore                   # Excluye .onnx, csv, credenciales
├── .dockerignore
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI - endpoints
│   ├── inference.py             # Inferencia con ONNX Runtime
│   ├── schemas.py               # Pydantic models
│   └── prediction_logger.py     # Registro en GCS (con fallback local)
├── tests/
│   ├── __init__.py
│   ├── test_inference.py        # Test del modelo
│   ├── test_api.py              # Test de endpoints
│   └── test_metrics.py          # Test del umbral de métrica
├── scripts/
│   ├── train_export_onnx.py     # Entrenamiento + exportación ONNX
│   ├── download_model.py        # Descarga desde GCS
│   ├── download_test_data.py    # Descarga desde GCS
│   └── validate_model.py        # Validación standalone
├── config/
│   ├── dev.env.example
│   └── prod.env.example
├── docs/
│   ├── demo_plan.md             # Guion de la demo de 10 min
│   └── decisiones_tecnicas.md   # ADR técnicas
└── .github/
    └── workflows/
        └── ci-cd.yml            # Pipeline CI/CD
```

---

## 16. Setup inicial completo (paso a paso)

### Pre-requisitos
- Cuenta en Google Cloud Platform con un proyecto
- gcloud CLI instalado y autenticado
- Docker Desktop (para builds locales)
- Python 3.11+

### 1. Habilitar APIs

```bash
gcloud services enable \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  run.googleapis.com
```

### 2. Crear Service Account

```bash
gcloud iam service-accounts create mlops-ci-cd \
  --display-name="MLOps CI/CD"

PROJECT_ID=$(gcloud config get-value project)
SA_EMAIL="mlops-ci-cd@${PROJECT_ID}.iam.gserviceaccount.com"

# Asignar roles
for role in storage.objectAdmin artifactregistry.writer run.admin iam.serviceAccountUser; do
  gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/$role"
done

# Generar key (subir a GitHub Secret GCP_SERVICE_ACCOUNT_KEY)
gcloud iam service-accounts keys create gcp-key.json \
  --iam-account=$SA_EMAIL
```

### 3. Crear Artifact Registry

```bash
gcloud artifacts repositories create mlops-enfermedad \
  --repository-format=docker \
  --location=us-central1 \
  --description="Imágenes Docker MLOps"
```

### 4. Crear buckets

```bash
for env in dev prod; do
  gcloud storage buckets create gs://icesi-mlops-models-$env --location=us-central1
  gcloud storage buckets create gs://icesi-mlops-data-$env   --location=us-central1
  gcloud storage buckets create gs://icesi-mlops-logs-$env   --location=us-central1
done
```

### 5. Entrenar y subir artefactos

```bash
python scripts/train_export_onnx.py
gcloud storage cp artifacts/modelo_enfermedad.onnx gs://icesi-mlops-models-dev/
gcloud storage cp artifacts/modelo_enfermedad.onnx gs://icesi-mlops-models-prod/
gcloud storage cp artifacts/test_data.csv          gs://icesi-mlops-data-dev/
gcloud storage cp artifacts/test_data.csv          gs://icesi-mlops-data-prod/
```

### 6. Configurar GitHub Secrets

En tu repo → Settings → Secrets and variables → Actions, agrega:
- `GCP_SERVICE_ACCOUNT_KEY` (contenido completo de gcp-key.json)
- `GCP_PROJECT_ID`
- Los 6 nombres de buckets (`GCS_*_BUCKET_DEV/PROD`)

### 7. Crear las ramas

```bash
git checkout -b dev && git push -u origin dev
git checkout -b prod && git push -u origin prod
```

¡Listo! Cualquier push a estas ramas dispara el CI/CD.

---

## Autores

Cristian Camilo Quebrada · Ruben Dario Sabogal · Edwin Perez L
Universidad Icesi — Maestría en IA Aplicada — 2026

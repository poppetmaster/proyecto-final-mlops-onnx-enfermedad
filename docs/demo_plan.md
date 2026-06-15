# Plan de Demo — 10 minutos

**Proyecto Final MLOps · Universidad Icesi**

---

## Objetivo

Demostrar que el pipeline MLOps completo funciona end-to-end: desde un push
de código en GitHub hasta la predicción en producción con logs verificables
en Google Cloud Storage.

---

## Pre-demo (5 min antes)

- [ ] Tener pestañas abiertas en el navegador:
  - Repositorio GitHub (rama `dev`)
  - GitHub Actions (último workflow run)
  - GCP Console — Cloud Run services
  - GCP Console — Artifact Registry (repo de imágenes)
  - GCP Console — GCS buckets (los 6 buckets)
- [ ] Terminal abierta con gcloud autenticado
- [ ] Tener listo el JSON de prueba en un archivo o portapapeles

```json
{
  "num_sintomas": 8,
  "dias_sintomas": 15,
  "nivel_dolor": 7,
  "tiene_fiebre": 1,
  "enfermedad_base": 0,
  "edad": 50
}
```

---

## Guion (10 minutos)

### Minuto 1 — Visión general y repositorio

> "Construimos un pipeline MLOps completo para una API que clasifica el estado
> de enfermedad de un paciente. El modelo está en formato ONNX, vive en GCS,
> y se despliega en Cloud Run con dos ambientes."

**Mostrar:**
- Estructura del repo en GitHub
- Las dos ramas: `dev` y `prod`
- README.md con la arquitectura

### Minuto 2 — Pipeline CI/CD

**Mostrar GitHub Actions:**
- `ci-cd.yml` con sus dos jobs: `test` y `build-promote`
- Último workflow run exitoso
- Logs del job `test` mostrando:
  - Descarga del modelo desde GCS
  - Descarga de test data desde GCS
  - Ejecución de pytest
  - Validación de métrica (F1 ≥ 0.80)

### Minuto 3 — GCS y artefactos

**Mostrar en GCS:**
- Bucket de modelos: `icesi-mlops-models-dev/modelo_enfermedad.onnx`
- Bucket de datos: `icesi-mlops-data-dev/test_data.csv`
- Bucket de logs: `icesi-mlops-logs-dev/predicciones_dev.txt`

> "Ni el modelo ONNX ni los datos están en el repo. Se descargan dinámicamente
> durante el pipeline. Esto es importante porque permite versionar el modelo
> independientemente del código."

### Minuto 4 — Artifact Registry y Cloud Run

**Mostrar:**
- Imágenes en Artifact Registry con tags `dev`, `prod` y SHA del commit
- Servicios en Cloud Run: `mlops-enfermedad-dev` y `mlops-enfermedad-prod`
- URLs de cada servicio
- Variables de entorno inyectadas

### Minuto 5 — Probar el endpoint DEV

```bash
# Obtener URL
DEV_URL=$(gcloud run services describe mlops-enfermedad-dev \
  --region=us-central1 --format='value(status.url)')

# Health check
curl $DEV_URL/health

# Info del modelo
curl $DEV_URL/model-info | jq

# Predicción
curl -X POST $DEV_URL/predict \
  -H "Content-Type: application/json" \
  -d @prediction.json | jq
```

**Mostrar respuesta:** environment, prediction, prediction_id, model_version, timestamp.

### Minuto 6 — Probar el endpoint PROD

```bash
PROD_URL=$(gcloud run services describe mlops-enfermedad-prod \
  --region=us-central1 --format='value(status.url)')

curl -X POST $PROD_URL/predict \
  -H "Content-Type: application/json" \
  -d @prediction.json | jq
```

> "La respuesta es idéntica en estructura, pero observa: `environment=prod` y
> `model_version` distinta. Son ambientes completamente independientes."

### Minuto 7 — Verificar logs en GCS

```bash
# Ver últimas líneas del log dev
gcloud storage cat gs://icesi-mlops-logs-dev/predicciones_dev.txt | tail -5

# Ver últimas líneas del log prod
gcloud storage cat gs://icesi-mlops-logs-prod/predicciones_prod.txt | tail -5
```

> "Cada predicción queda registrada con su UUID, timestamp, input y output.
> Esto nos da trazabilidad completa para auditoría y futuro reentrenamiento."

### Minuto 8 — Demostrar el ciclo dev → prod

**Hacer un cambio trivial en `dev`:**

```bash
git checkout dev
# Editar README o cambiar MODEL_VERSION
git commit -am "demo: trigger pipeline"
git push origin dev
```

**Mostrar en GitHub Actions:** el workflow arrancando automáticamente.

> "Cualquier push a dev dispara el pipeline. Si los tests pasan, se construye
> la imagen y se despliega automáticamente a Cloud Run DEV. Para promover a
> producción simplemente hacemos merge a la rama prod."

### Minuto 9 — Resumen de capacidades MLOps cubiertas

> "Resumen de lo que cubre este proyecto:"

- ✅ **Versionado**: modelo en GCS, código en Git, imágenes en Artifact Registry
- ✅ **Tests automatizados**: unitarios + métrica con umbral mínimo
- ✅ **CI/CD por rama**: dev y prod independientes
- ✅ **Despliegue serverless**: Cloud Run con autoscaling
- ✅ **Trazabilidad**: cada predicción registrada con UUID y timestamp
- ✅ **Formato portable**: ONNX permite cambiar de framework sin tocar el runtime
- ✅ **Separación de configuración**: variables de entorno, secrets

### Minuto 10 — Preguntas y limitaciones

**Limitaciones que reconocemos:**
- Append en GCS no es eficiente para alto volumen (mencionar Pub/Sub + BigQuery)
- Cold starts en Cloud Run (mitigable con min-instances)
- No incluye reentrenamiento automático ni detección de drift

**Próximos pasos sugeridos:**
- Integrar Cloud Logging para los logs de predicción
- Agregar detector de drift de datos (Evidently AI)
- Reentrenamiento programado con Cloud Composer / Airflow

---

## Comandos rápidos para tener a la mano

```bash
# URLs
DEV_URL=$(gcloud run services describe mlops-enfermedad-dev --region=us-central1 --format='value(status.url)')
PROD_URL=$(gcloud run services describe mlops-enfermedad-prod --region=us-central1 --format='value(status.url)')

# Test rápido
curl -X POST $DEV_URL/predict -H "Content-Type: application/json" -d '{"num_sintomas":8,"dias_sintomas":15,"nivel_dolor":7,"tiene_fiebre":1,"enfermedad_base":0,"edad":50}' | jq

# Ver logs
gcloud storage cat gs://icesi-mlops-logs-dev/predicciones_dev.txt | tail -10

# Ver imagen más reciente
gcloud artifacts docker images list us-central1-docker.pkg.dev/$(gcloud config get-value project)/mlops-enfermedad --limit=5
```

---

## Plan de contingencia

**Si Cloud Run no responde:**
- Mostrar localmente con `docker run` el contenedor que está en Artifact Registry
- Comando: `docker pull <imagen>` y `docker run -p 8080:8080 <imagen>`

**Si el push de dev no dispara el workflow:**
- Mostrar la última ejecución exitosa
- Explicar el flujo con los logs en GitHub Actions

**Si GCS no muestra el log:**
- Verificar permisos del Service Account
- Mostrar el modo fallback local en `/tmp/predicciones_dev.txt` dentro del contenedor

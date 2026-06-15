# Decisiones Técnicas (ADR)

Documento que recoge las decisiones técnicas relevantes del proyecto y la
justificación detrás de cada una.

---

## ADR-001 · ONNX como formato del modelo

**Decisión:** Usar ONNX (Open Neural Network Exchange) como formato de
serialización del modelo de ML.

**Alternativas evaluadas:**
- pickle / joblib (formato nativo de sklearn)
- TensorFlow SavedModel
- TorchScript

**Justificación:**
- **Portabilidad:** ONNX es un estándar abierto soportado por sklearn, PyTorch,
  TensorFlow, XGBoost, etc. Permite cambiar el framework de entrenamiento sin
  modificar el código de inferencia.
- **Sin dependencias de sklearn en runtime:** la imagen Docker final no
  necesita sklearn, solo `onnxruntime`. Esto reduce el tamaño y la superficie
  de ataque.
- **Performance:** `onnxruntime` está optimizado a nivel C++ y suele superar
  la inferencia con sklearn puro.
- **Seguridad:** pickle es inseguro (permite ejecución arbitraria de código);
  ONNX es solo un formato binario de grafo computacional.

**Trade-offs:**
- Algunos modelos sklearn complejos no se exportan perfectamente.
- Requiere un paso adicional de conversión durante el entrenamiento.

---

## ADR-002 · Cloud Run como plataforma de despliegue

**Decisión:** Desplegar la API en Google Cloud Run.

**Alternativas evaluadas:**
- GKE (Google Kubernetes Engine)
- Cloud Functions
- App Engine
- VM Compute Engine

**Justificación:**
- **Serverless:** escala de 0 a N automáticamente. Sin servidores que mantener.
- **Pago por uso:** ideal para una demo y para tráfico variable.
- **Soporte nativo de Docker:** corre cualquier contenedor que escuche en `$PORT`.
- **Integración directa con Artifact Registry y GCS:** misma plataforma GCP.
- **Soporte gradual de tráfico:** permite blue/green y canary deployments.
- **HTTPS automático:** sin certificados que gestionar.

**Trade-offs:**
- Cold starts (~3-5s) si `min-instances=0`. Mitigable subiendo `min-instances`.
- Timeout máximo de 60 minutos por request (más que suficiente para una API).

---

## ADR-003 · Artefactos versionados en GCS, no en Git

**Decisión:** El modelo `.onnx` y los datos de prueba `test_data.csv` viven
en buckets de GCS y NO están en el repositorio.

**Justificación:**
- **Tamaño:** Git no está diseñado para binarios grandes. Aunque el modelo es
  pequeño en este proyecto, en escenarios reales pueden llegar a GB.
- **Versionado independiente:** el modelo puede actualizarse sin necesitar
  un commit al repo.
- **Separación de responsabilidades:** el código vive en Git, los artefactos
  en almacenamiento de objetos.
- **Estándar de la industria:** prácticamente todas las herramientas MLOps
  (MLflow, DVC, Vertex AI) siguen este patrón.

**Trade-offs:**
- Hay que orquestar la descarga durante el CI/CD.
- Requiere credenciales GCP para reproducir localmente.

---

## ADR-004 · Pipeline CI/CD por rama (dev/prod)

**Decisión:** El pipeline se dispara por push a las ramas `dev` o `prod`,
y selecciona automáticamente los buckets y servicios correspondientes.

**Alternativas evaluadas:**
- Una sola rama `main` con deploy manual a cada entorno
- Tags de Git para los releases a prod
- GitFlow completo con `develop`, `feature/*`, `release/*`

**Justificación:**
- **Simplicidad:** dos ramas, dos ambientes. Fácil de entender y enseñar.
- **Trazabilidad clara:** se sabe qué hay en cada entorno mirando la rama.
- **Promoción explícita:** hacer merge de `dev` → `prod` es un acto deliberado.
- **Suficiente para MVP:** GitFlow sería overkill para el alcance del proyecto.

**Trade-offs:**
- No hay rollback automático si prod falla; requiere intervención manual.
- Para equipos grandes se necesitaría algo más sofisticado (release branches).

---

## ADR-005 · FastAPI como framework de la API

**Decisión:** Usar FastAPI para exponer el modelo.

**Alternativas evaluadas:**
- Flask
- Django REST Framework
- BentoML
- TorchServe / TensorFlow Serving

**Justificación:**
- **Performance:** basado en Starlette + asyncio, uno de los frameworks más
  rápidos en Python.
- **Validación automática:** Pydantic valida entradas sin código adicional.
- **Documentación automática:** OpenAPI/Swagger en `/docs` sin esfuerzo.
- **Tipado moderno:** type hints nativos, mejor mantenibilidad.
- **Curva de aprendizaje:** mínima para quien ya conoce Flask.

---

## ADR-006 · RandomForestClassifier como modelo base

**Decisión:** Usar `RandomForestClassifier` de sklearn.

**Justificación:**
- **Buena precisión sobre datos tabulares:** estado del arte sin tuning.
- **Sin requerir escalamiento de features:** robusto a valores en diferentes escalas.
- **Interpretable:** importancia de features fácil de calcular (útil para validación clínica).
- **Conversión limpia a ONNX:** `skl2onnx` lo soporta perfectamente.

**Trade-offs:**
- Modelo más pesado que una regresión logística.
- No es el más rápido en inferencia. Para latencia ultra-baja, una red neuronal
  pequeña sería preferible.

---

## ADR-007 · Métrica F1 weighted con umbral 0.80

**Decisión:** Usar F1 weighted como métrica primaria de validación, con
umbral mínimo 0.80 para dev y 0.85 recomendado para prod.

**Justificación:**
- **F1 vs Accuracy:** F1 es mejor cuando hay clases desbalanceadas, lo cual es
  típico en datasets médicos (más pacientes "sanos" que "terminales").
- **Weighted:** pondera por el tamaño de cada clase, evitando que clases
  raras "tiren" la métrica hacia abajo.
- **Umbral 0.80:** balance razonable entre exigencia y factibilidad. En
  producción real se subiría el umbral conforme se valide el modelo.

---

## ADR-008 · Logs de predicciones en GCS (con fallback local)

**Decisión:** Cada predicción se registra como una línea JSON en un archivo
TXT dentro de un bucket de GCS. Si GCS falla, se hace fallback a un archivo
local dentro del contenedor.

**Alternativas evaluadas:**
- Cloud Logging (Stackdriver)
- BigQuery
- Pub/Sub + sistema de procesamiento downstream

**Justificación:**
- **Simplicidad:** un archivo TXT es trivial de inspeccionar y procesar.
- **Cumple con el requerimiento del proyecto:** el enunciado pide TXT en GCS.
- **Fallback robusto:** la API nunca falla por un error de logging.

**Trade-offs:**
- **Append no es nativo en GCS:** cada escritura descarga, concatena y sube.
  No es escalable para tráfico alto. Mitigación: rotar archivos por hora/día,
  o pasar a Cloud Logging en producción real.

---

## ADR-009 · Python 3.11-slim como imagen base

**Decisión:** Usar `python:3.11-slim` como imagen Docker base.

**Justificación:**
- **Compatibilidad:** Python 3.11 tiene soporte completo de todas las libs.
- **Performance:** 3.11 trae mejoras significativas vs 3.10 (~10-25%).
- **Tamaño:** la variante `slim` es ~120MB, mucho menor que la imagen completa.
- **Cloud Run compatible:** corre sin problemas en serverless.

**Trade-offs:**
- `slim` no trae compiladores; para libs con bindings nativos puede requerir
  instalar `build-essential`. En este proyecto no hace falta porque las wheels
  precompiladas existen para todas las dependencias.

---

## Resumen de stack tecnológico

| Capa | Tecnología | Versión |
|---|---|---|
| Lenguaje | Python | 3.11 |
| API Framework | FastAPI | latest |
| Servidor ASGI | Uvicorn | latest |
| Inferencia | ONNX Runtime | latest |
| Entrenamiento | scikit-learn + skl2onnx | latest |
| Validación | Pydantic | 2.x |
| Containerización | Docker | - |
| Container Registry | Artifact Registry | - |
| Compute | Cloud Run | - |
| Storage | Google Cloud Storage | - |
| CI/CD | GitHub Actions | - |
| Testing | pytest + httpx | latest |

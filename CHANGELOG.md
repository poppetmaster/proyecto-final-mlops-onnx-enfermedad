# Changelog

Todas las modificaciones notables al proyecto se documentan aquí.
El formato sigue [Keep a Changelog](https://keepachangelog.com/) y este
proyecto adhiere a [Semantic Versioning](https://semver.org/).

---

## [1.0.0] - 2026-06-14

### Added — Proyecto Final MLOps
- Pipeline CI/CD completo con GitHub Actions (`test` + `build-promote`)
- Soporte para ramas `dev` y `prod` con buckets y servicios independientes
- Modelo en formato ONNX exportado desde RandomForestClassifier (skl2onnx)
- Descarga del modelo y datos de prueba desde Google Cloud Storage
- Validación de métrica F1 weighted ≥ 0.80 en CI antes de promover
- Despliegue automatizado a Cloud Run (dev/prod)
- Imágenes Docker publicadas en Artifact Registry
- Registro de cada predicción en GCS con fallback local
- Tests pytest: inferencia, API, métricas
- Documentación completa: README, plan de demo, decisiones técnicas

### Architecture
- FastAPI + ONNX Runtime (sin sklearn en runtime)
- GCS para artefactos (modelo, datos, logs)
- Service Account con permisos mínimos necesarios
- Health check para Cloud Run en `/health`

---

## [0.2.0] - 2026-05-13 — Taller 2

### Added
- Quinta categoría: `ENFERMEDAD TERMINAL`
- Endpoint `/stats` con estadísticas de predicciones
- Registro de predicciones en archivo JSON local
- 15 pruebas unitarias con pytest
- GitHub Actions básico

---

## [0.1.0] - 2026-05-06 — Taller 1

### Added
- API FastAPI con función de predicción simulada
- 4 estados: NO ENFERMO, LEVE, AGUDA, CRÓNICA
- Interfaz web con formulario para el médico
- Endpoint API REST `/api/predecir`
- Dockerfile compatible con Windows
- README con instrucciones de build y run

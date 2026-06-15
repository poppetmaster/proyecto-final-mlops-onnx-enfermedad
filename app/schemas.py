"""
Esquemas Pydantic para validación de entrada/salida de la API.

Universidad Icesi · Maestría en IA Aplicada · Proyecto Final MLOps
"""

from pydantic import BaseModel, Field
from typing import Optional


class PatientInput(BaseModel):
    """Datos clínicos del paciente para predicción."""

    num_sintomas: int = Field(
        ..., ge=0, le=20,
        description="Número de síntomas reportados (0-20)",
        json_schema_extra={"example": 5},
    )
    dias_sintomas: int = Field(
        ..., ge=0, le=365,
        description="Días con síntomas activos (0-365)",
        json_schema_extra={"example": 7},
    )
    nivel_dolor: int = Field(
        ..., ge=0, le=10,
        description="Nivel de dolor en escala 0-10",
        json_schema_extra={"example": 6},
    )
    tiene_fiebre: int = Field(
        default=0, ge=0, le=1,
        description="¿Presenta fiebre? (0=no, 1=sí)",
    )
    enfermedad_base: int = Field(
        default=0, ge=0, le=1,
        description="¿Tiene enfermedad preexistente? (0=no, 1=sí)",
    )
    edad: int = Field(
        default=30, ge=0, le=120,
        description="Edad del paciente en años",
    )


class PredictionResponse(BaseModel):
    """Respuesta de la API con la predicción y metadatos."""

    environment: str = Field(..., description="Entorno (dev/prod/local)")
    prediction: str = Field(..., description="Etiqueta del estado predicho")
    prediction_class: int = Field(..., description="Clase numérica predicha 0-4")
    prediction_id: str = Field(..., description="UUID único de la predicción")
    model_version: str = Field(..., description="Versión del modelo")
    timestamp: str = Field(..., description="Fecha-hora UTC ISO 8601")
    input: dict = Field(..., description="Datos de entrada recibidos")


class HealthResponse(BaseModel):
    """Respuesta del endpoint /health."""

    status: str
    environment: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    """Información del modelo cargado."""

    model_version: str
    model_path: str
    input_features: list
    output_classes: list
    environment: str

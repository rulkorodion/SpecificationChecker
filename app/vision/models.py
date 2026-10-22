"""Результат визуального детектора."""

from __future__ import annotations

from pydantic import Field

from app.enums import PresenceStatus
from app.schemas import StrictModel


class VisualDetection(StrictModel):
    status: PresenceStatus
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    candidate_bbox: list[int] | None = None
    candidate_count: int = Field(default=0, ge=0)
    features: dict[str, float | int | bool | str] = Field(default_factory=dict)
    profile_scores: dict[str, float] = Field(default_factory=dict)

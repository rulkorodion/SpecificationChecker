"""SQLAlchemy 2.0 models for specification-check jobs."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    DateTime,
    Enum as SQLAlchemyEnum,
    Float,
    Index,
    JSON,
    MetaData,
    String,
    Text,
    Integer,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class JobStatus(StrEnum):
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class OneCDeliveryStatus(StrEnum):
    PENDING = "PENDING"
    SENT = "SENT"


NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


JSON_PAYLOAD = JSON().with_variant(JSONB(), "postgresql")


class CheckJob(Base):
    __tablename__ = "specification_checks"
    __table_args__ = (
        Index("ix_specification_checks_external_id", "external_id"),
        Index("ix_specification_checks_status", "status"),
        Index("ix_specification_checks_created_at", "created_at"),
    )

    job_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    external_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    status: Mapped[JobStatus] = mapped_column(
        SQLAlchemyEnum(
            JobStatus,
            name="check_job_status",
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            length=16,
        ),
        nullable=False,
    )
    decision: Mapped[str | None] = mapped_column(String(32), nullable=True)
    comment_1c: Mapped[str | None] = mapped_column(Text, nullable=True)
    result: Mapped[dict[str, Any] | None] = mapped_column(
        JSON_PAYLOAD, nullable=True
    )
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_details: Mapped[dict[str, Any] | None] = mapped_column(
        JSON_PAYLOAD, nullable=True
    )
    document_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    document_sha256: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    processing_time_seconds: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    onec_delivery_status: Mapped[OneCDeliveryStatus | None] = mapped_column(
        SQLAlchemyEnum(
            OneCDeliveryStatus,
            name="onec_delivery_status",
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            length=16,
        ),
        nullable=True,
    )
    onec_delivery_attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    onec_payload: Mapped[dict[str, Any] | None] = mapped_column(
        JSON_PAYLOAD, nullable=True
    )
    onec_receipt: Mapped[dict[str, Any] | None] = mapped_column(
        JSON_PAYLOAD, nullable=True
    )
    onec_last_error: Mapped[dict[str, Any] | None] = mapped_column(
        JSON_PAYLOAD, nullable=True
    )
    onec_delivery_updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

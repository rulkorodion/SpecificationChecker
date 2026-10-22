"""Persistent job storage for the HTTP API."""

from app.persistence.database import Database, DatabaseConfigurationError
from app.persistence.models import Base, CheckJob, JobStatus, OneCDeliveryStatus
from app.persistence.repository import CheckJobRecord, CheckJobRepository
from app.persistence.service import JobPersistenceError, JobPersistenceService

__all__ = [
    "Base",
    "CheckJob",
    "CheckJobRecord",
    "CheckJobRepository",
    "Database",
    "DatabaseConfigurationError",
    "JobPersistenceService",
    "JobPersistenceError",
    "JobStatus",
    "OneCDeliveryStatus",
]

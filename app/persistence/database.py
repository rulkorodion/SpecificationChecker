"""SQLAlchemy engine/session lifecycle configured through DATABASE_URL."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool


class DatabaseConfigurationError(RuntimeError):
    """The server cannot create a database connection from its environment."""


def normalize_database_url(url: str) -> str:
    """Prefer Psycopg 3 for common PostgreSQL URL spellings."""

    normalized = url.strip()
    if normalized.startswith("postgres://"):
        return "postgresql+psycopg://" + normalized.removeprefix("postgres://")
    if normalized.startswith("postgresql://"):
        return "postgresql+psycopg://" + normalized.removeprefix(
            "postgresql://"
        )
    return normalized


class Database:
    def __init__(self, database_url: str, *, engine: Engine | None = None) -> None:
        if not database_url.strip():
            raise DatabaseConfigurationError("DATABASE_URL не задан")
        self.url = normalize_database_url(database_url)
        if engine is None:
            options: dict = {"pool_pre_ping": True}
            if self.url.startswith("sqlite"):
                options["connect_args"] = {"check_same_thread": False}
                if self.url in {"sqlite://", "sqlite:///:memory:"}:
                    options["poolclass"] = StaticPool
            engine = create_engine(self.url, **options)
        self.engine = engine
        self.session_factory = sessionmaker(
            bind=self.engine,
            class_=Session,
            expire_on_commit=False,
            autoflush=False,
        )

    @classmethod
    def from_environment(cls) -> "Database":
        value = os.getenv("DATABASE_URL", "")
        return cls(value)

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self.session_factory() as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

    def ping(self) -> None:
        with self.engine.connect() as connection:
            connection.execute(text("SELECT 1"))

    def dispose(self) -> None:
        self.engine.dispose()

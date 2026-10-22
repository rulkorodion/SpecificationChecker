"""Persist 1C result delivery state in PostgreSQL.

Revision ID: 20260804_0002
Revises: 20260804_0001
Create Date: 2026-08-04
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260804_0002"
down_revision: str | None = "20260804_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "specification_checks",
        sa.Column("onec_delivery_status", sa.String(length=16), nullable=True),
    )
    op.add_column(
        "specification_checks",
        sa.Column(
            "onec_delivery_attempts",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
    )
    op.add_column(
        "specification_checks",
        sa.Column("onec_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column(
        "specification_checks",
        sa.Column("onec_receipt", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column(
        "specification_checks",
        sa.Column("onec_last_error", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column(
        "specification_checks",
        sa.Column("onec_delivery_updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_check_constraint(
        op.f("ck_specification_checks_onec_delivery_status"),
        "specification_checks",
        "onec_delivery_status IS NULL OR onec_delivery_status IN ('PENDING', 'SENT')",
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("ck_specification_checks_onec_delivery_status"),
        "specification_checks",
        type_="check",
    )
    for column in (
        "onec_delivery_updated_at",
        "onec_last_error",
        "onec_receipt",
        "onec_payload",
        "onec_delivery_attempts",
        "onec_delivery_status",
    ):
        op.drop_column("specification_checks", column)

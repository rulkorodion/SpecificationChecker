"""Create persistent specification check jobs.

Revision ID: 20260804_0001
Revises:
Create Date: 2026-08-04
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260804_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "specification_checks",
        sa.Column("job_id", sa.String(length=36), nullable=False),
        sa.Column("external_id", sa.String(length=128), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("decision", sa.String(length=32), nullable=True),
        sa.Column("comment_1c", sa.Text(), nullable=True),
        sa.Column("result", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "error_details",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("document_path", sa.Text(), nullable=True),
        sa.Column("document_sha256", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("processing_time_seconds", sa.Float(), nullable=True),
        sa.CheckConstraint(
            "status IN ('PROCESSING', 'COMPLETED', 'FAILED')",
            name="check_job_status",
        ),
        sa.PrimaryKeyConstraint("job_id", name="pk_specification_checks"),
    )
    op.create_index(
        "ix_specification_checks_created_at",
        "specification_checks",
        ["created_at"],
    )
    op.create_index(
        "ix_specification_checks_external_id",
        "specification_checks",
        ["external_id"],
    )
    op.create_index(
        "ix_specification_checks_status",
        "specification_checks",
        ["status"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_specification_checks_status", table_name="specification_checks"
    )
    op.drop_index(
        "ix_specification_checks_external_id",
        table_name="specification_checks",
    )
    op.drop_index(
        "ix_specification_checks_created_at",
        table_name="specification_checks",
    )
    op.drop_table("specification_checks")

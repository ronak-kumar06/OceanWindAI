"""Initial schema — analysis_tasks, wind_results, validation_results

Revision ID: 001
Revises:
Create Date: 2026-06-27

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "analysis_tasks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("date_selected", sa.String(), nullable=False),
        sa.Column("bbox_str", sa.String(), nullable=True),
        sa.Column("aoi_geojson", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=True),
        sa.Column("data_source", sa.String(), nullable=True),
        sa.Column("image_url", sa.String(), nullable=True),
        sa.Column("mean_speed", sa.Float(), nullable=True),
        sa.Column("dominant_dir", sa.Float(), nullable=True),
        sa.Column("n_vectors", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_analysis_tasks_id"), "analysis_tasks", ["id"], unique=False)

    op.create_table(
        "wind_results",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("task_id", sa.Integer(), nullable=True),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.Column("speed", sa.Float(), nullable=False),
        sa.Column("direction", sa.Float(), nullable=False),
        sa.Column("u_component", sa.Float(), nullable=False),
        sa.Column("v_component", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(["task_id"], ["analysis_tasks.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_wind_results_id"), "wind_results", ["id"], unique=False)

    op.create_table(
        "validation_results",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("task_id", sa.Integer(), nullable=True),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("rmse_speed", sa.Float(), nullable=True),
        sa.Column("mae_speed", sa.Float(), nullable=True),
        sa.Column("bias_speed", sa.Float(), nullable=True),
        sa.Column("corr_speed", sa.Float(), nullable=True),
        sa.Column("rmse_dir", sa.Float(), nullable=True),
        sa.Column("bias_dir", sa.Float(), nullable=True),
        sa.Column("n_samples", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["task_id"], ["analysis_tasks.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_validation_results_id"), "validation_results", ["id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_validation_results_id"), table_name="validation_results")
    op.drop_table("validation_results")
    op.drop_index(op.f("ix_wind_results_id"), table_name="wind_results")
    op.drop_table("wind_results")
    op.drop_index(op.f("ix_analysis_tasks_id"), table_name="analysis_tasks")
    op.drop_table("analysis_tasks")

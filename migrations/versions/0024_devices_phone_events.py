"""Phone bridge: devices, phone_events, device_commands.

Revision ID: 0024
Revises: 0023
Create Date: 2026-11-25

pawn-assistant (Android) registers as a device, pushes opt-in context
(calls, SMS, notifications, location, status) and runs phone tools queued
by pawn-agent. See docs/PHONE_BRIDGE.md.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0024"
down_revision: Union[str, None] = "0023"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "devices",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("token_hash", sa.String(), nullable=False),
        sa.Column("unifiedpush_endpoint", sa.Text(), nullable=True),
        sa.Column("enabled_categories", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("last_seen", sa.DateTime(), nullable=True),
    )
    op.create_table(
        "phone_events",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "device_id",
            sa.String(),
            sa.ForeignKey("devices.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("ts", sa.DateTime(), nullable=False),
        sa.Column("dedup_key", sa.String(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("received_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint("device_id", "dedup_key", name="uq_phone_events_dedup"),
    )
    op.create_index("ix_phone_events_device_id", "phone_events", ["device_id"])
    op.create_index("ix_phone_events_kind", "phone_events", ["kind"])
    op.create_index("ix_phone_events_ts", "phone_events", ["ts"])
    op.create_table(
        "device_commands",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "device_id",
            sa.String(),
            sa.ForeignKey("devices.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("tool", sa.String(), nullable=False),
        sa.Column("args", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_device_commands_device_id", "device_commands", ["device_id"])
    op.create_index("ix_device_commands_status", "device_commands", ["status"])


def downgrade() -> None:
    op.drop_index("ix_device_commands_status", table_name="device_commands")
    op.drop_index("ix_device_commands_device_id", table_name="device_commands")
    op.drop_table("device_commands")
    op.drop_index("ix_phone_events_ts", table_name="phone_events")
    op.drop_index("ix_phone_events_kind", table_name="phone_events")
    op.drop_index("ix_phone_events_device_id", table_name="phone_events")
    op.drop_table("phone_events")
    op.drop_table("devices")

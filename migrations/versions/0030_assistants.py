"""Assistants: device kind/presence, voice threads, access requests, wake clips.

Revision ID: 0030
Revises: 0029
Create Date: 2026-11-27

Linux and Android voice assistants share one device registry (``devices``).
``kind`` separates phones from desktops (desktops take commands over an SSE
stream instead of UnifiedPush) and ``presence`` records what the assistant last
reported (active / idle / away, headset, call). ``voice_threads`` tracks each
``voice:`` conversation so another device can take it over.
``access_requests`` are phone-approved full-access grants for another device.
``wake_clips`` / ``wake_models`` back shared wake-word training. See
docs/ASSISTANTS.md.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0030"
down_revision: Union[str, None] = "0029"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "devices", sa.Column("kind", sa.String(), nullable=False, server_default="phone")
    )
    op.add_column("devices", sa.Column("platform", sa.String(), nullable=True))
    op.add_column("devices", sa.Column("presence", sa.JSON(), nullable=True))
    op.add_column("devices", sa.Column("presence_at", sa.DateTime(), nullable=True))

    op.create_table(
        "voice_threads",
        sa.Column("conversation", sa.String(), primary_key=True),
        sa.Column(
            "device_id",
            sa.String(),
            sa.ForeignKey("devices.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        ),
        sa.Column("access", sa.String(), nullable=False, server_default="chat"),
        sa.Column("lang", sa.String(), nullable=False, server_default="it"),
        sa.Column("state", sa.String(), nullable=False, server_default="active", index=True),
        sa.Column("turns", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_user", sa.Text(), nullable=True),
        sa.Column("last_answer", sa.Text(), nullable=True),
        sa.Column("portal_chat_id", sa.String(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("last_turn_at", sa.DateTime(), nullable=True, index=True),
    )

    op.create_table(
        "access_requests",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "requester_device_id",
            sa.String(),
            sa.ForeignKey("devices.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "approver_device_id",
            sa.String(),
            sa.ForeignKey("devices.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("conversation", sa.String(), nullable=True),
        sa.Column("minutes", sa.Integer(), nullable=False, server_default="15"),
        sa.Column("status", sa.String(), nullable=False, server_default="pending", index=True),
        sa.Column("command_id", sa.String(), nullable=True),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("access_until", sa.DateTime(), nullable=True),
    )

    op.create_table(
        "wake_clips",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "device_id",
            sa.String(),
            sa.ForeignKey("devices.id", ondelete="CASCADE"),
            nullable=True,
            index=True,
        ),
        sa.Column("label", sa.String(), nullable=False, index=True),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("path", sa.String(), nullable=False),
        sa.Column("size", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("meta", sa.JSON(), nullable=True),
        sa.Column("captured_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True, index=True),
    )

    op.create_table(
        "wake_models",
        sa.Column("version", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False, server_default="hey_pawn"),
        sa.Column("path", sa.String(), nullable=False),
        sa.Column("sha256", sa.String(), nullable=False),
        sa.Column("size", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("thresholds", sa.JSON(), nullable=True),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("wake_models")
    op.drop_table("wake_clips")
    op.drop_table("access_requests")
    op.drop_table("voice_threads")
    op.drop_column("devices", "presence_at")
    op.drop_column("devices", "presence")
    op.drop_column("devices", "platform")
    op.drop_column("devices", "kind")

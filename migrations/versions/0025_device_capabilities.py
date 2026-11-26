"""Phone bridge: live tool capabilities declared by each device.

Revision ID: 0025
Revises: 0024
Create Date: 2026-11-26

The phone declares which live tools it will run (``location_now``,
``sms_send`` …) based on its toggles and build flavor; the server only
queues commands for declared capabilities. See docs/PHONE_BRIDGE.md.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0025"
down_revision: Union[str, None] = "0024"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("devices", sa.Column("capabilities", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("devices", "capabilities")

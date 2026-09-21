"""Regression: shared SQLAlchemy engines must not leak connection pools."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pawn_core.database as database


def setup_function() -> None:
    database.dispose_engines()


def teardown_function() -> None:
    database.dispose_engines()


def test_get_engine_reuses_cached_engine_per_dsn() -> None:
    engines = [MagicMock(name="engine-a"), MagicMock(name="engine-b")]

    with patch.object(database, "create_engine", side_effect=engines) as create:
        first = database.get_engine("postgresql+psycopg://db/a")
        second = database.get_engine("postgresql+psycopg://db/a")
        other = database.get_engine("postgresql+psycopg://db/b")

    assert first is second
    assert first is engines[0]
    assert other is engines[1]
    assert create.call_count == 2
    create.assert_any_call("postgresql+psycopg://db/a", pool_pre_ping=True)
    create.assert_any_call("postgresql+psycopg://db/b", pool_pre_ping=True)


def test_get_session_reuses_engine_across_calls() -> None:
    engine = MagicMock(name="shared-engine")
    session_cm = MagicMock()
    session_cm.__enter__.return_value = MagicMock()
    session_cm.__exit__.return_value = False

    with (
        patch.object(database, "create_engine", return_value=engine) as create,
        patch.object(database, "Session", return_value=session_cm) as session_cls,
    ):
        with database._get_session("postgresql+psycopg://db/leak"):
            pass
        with database._get_session("postgresql+psycopg://db/leak"):
            pass

    assert create.call_count == 1
    assert session_cls.call_count == 2
    session_cls.assert_called_with(engine)
    engine.dispose.assert_not_called()


def test_dispose_engines_clears_cache() -> None:
    engines = [MagicMock(name="engine-1"), MagicMock(name="engine-2")]
    with patch.object(database, "create_engine", side_effect=engines) as create:
        first = database.get_engine("postgresql+psycopg://db/a")
        database.dispose_engines()
        second = database.get_engine("postgresql+psycopg://db/a")

    assert first is engines[0]
    assert second is engines[1]
    first.dispose.assert_called_once()
    assert create.call_count == 2

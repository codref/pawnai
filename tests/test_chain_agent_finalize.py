"""Recorder finalize policy: chain_agent resolution + already-processed paths."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from pawn_diarize.core.queue_listener import _pending_audio_paths, _resolve_chain_cfg


class TestResolveChainCfg:
    def test_false_skips_even_when_config_enabled(self):
        cfg = {"chain_agent": {"enabled": True, "command": "session_completed"}}
        assert _resolve_chain_cfg({"chain_agent": False}, cfg) is None
        assert _resolve_chain_cfg({"chain_agent": "false"}, cfg) is None

    def test_dict_finalize_chains_even_when_config_disabled(self):
        cfg = {"chain_agent": {"enabled": False, "command": "run", "prompt": "x"}}
        out = _resolve_chain_cfg(
            {"chain_agent": {"command": "session_completed"}}, cfg
        )
        assert out == {"prompt": "x", "command": "session_completed"}

    def test_none_falls_back_to_config(self):
        cfg = {
            "chain_agent": {
                "enabled": True,
                "command": "session_completed",
                "prompt": "Analyze.",
            }
        }
        out = _resolve_chain_cfg({"chain_agent": None}, cfg)
        assert out == {"prompt": "Analyze.", "command": "session_completed"}

    def test_none_skips_when_config_disabled(self):
        cfg = {"chain_agent": {"enabled": False}}
        assert _resolve_chain_cfg({}, cfg) is None


class TestPendingAudioPaths:
    def test_filters_already_processed(self):
        pending, skipped = _pending_audio_paths(
            ["s3://b/a.flac", "s3://b/b.flac"],
            ["s3://b/a.flac"],
        )
        assert pending == ["s3://b/b.flac"]
        assert skipped == 1

    def test_all_processed_yields_empty(self):
        pending, skipped = _pending_audio_paths(
            ["s3://b/a.flac"],
            ["s3://b/a.flac"],
        )
        assert pending == []
        assert skipped == 1

    def test_empty_processed_keeps_all(self):
        pending, skipped = _pending_audio_paths(["s3://b/a.flac"], [])
        assert pending == ["s3://b/a.flac"]
        assert skipped == 0


class TestSessionCompletedIdempotency:
    def _cfg(self):
        return SimpleNamespace(
            db_dsn="postgresql+psycopg://dummy/dummy",
            coworker=SimpleNamespace(enabled=True),
            chat_model_id="test-model",
        )

    def test_skips_when_same_segment_count_already_completed(self):
        from pawn_server.core import queue_listener as ql

        cfg = self._cfg()

        with (
            patch.object(ql, "_session_segment_count", return_value=12),
            patch.object(ql, "_already_session_completed", return_value=True),
            patch.object(ql, "_speakers_refresh", new_callable=AsyncMock) as refresh,
            patch(
                "pawn_agent.core.coworker.pipeline.process_session",
                new_callable=AsyncMock,
            ) as process,
        ):
            asyncio.run(ql._session_completed({"session_id": "meet-1"}, cfg))

        process.assert_not_awaited()
        refresh.assert_awaited_once()

    def test_runs_when_not_yet_covered(self):
        from pawn_server.core import queue_listener as ql

        cfg = self._cfg()

        with (
            patch.object(ql, "_session_segment_count", return_value=12),
            patch.object(ql, "_already_session_completed", return_value=False),
            patch.object(ql, "_speakers_refresh", new_callable=AsyncMock),
            patch(
                "pawn_agent.core.coworker.pipeline.process_session",
                new_callable=AsyncMock,
                return_value={"items": 2},
            ) as process,
            patch("pawn_agent.utils.db.create_agent_run", return_value="run-1"),
            patch("pawn_agent.utils.db.update_agent_run") as update,
        ):
            asyncio.run(ql._session_completed({"session_id": "meet-1"}, cfg))

        process.assert_awaited_once()
        assert update.call_count >= 2

    def test_force_bypasses_idempotency(self):
        from pawn_server.core import queue_listener as ql

        cfg = self._cfg()

        with (
            patch.object(ql, "_session_segment_count", return_value=12),
            patch.object(ql, "_already_session_completed", return_value=True) as already,
            patch.object(ql, "_speakers_refresh", new_callable=AsyncMock),
            patch(
                "pawn_agent.core.coworker.pipeline.process_session",
                new_callable=AsyncMock,
                return_value={"items": 1},
            ) as process,
            patch("pawn_agent.utils.db.create_agent_run", return_value="run-2"),
            patch("pawn_agent.utils.db.update_agent_run"),
        ):
            asyncio.run(
                ql._session_completed({"session_id": "meet-1", "force": True}, cfg)
            )

        already.assert_not_called()
        process.assert_awaited_once()


class TestTranscribeDiarizeSkipsProcessed:
    def test_finalize_republish_skips_transcription(self):
        from pawn_diarize.core.queue_listener import _run_transcribe_diarize

        params = {
            "audio_paths": ["s3://b/chunk-3.flac"],
            "session": "meet-1",
            "chain_agent": {"command": "session_completed"},
        }
        cfg = MagicMock()
        engine = MagicMock()

        with (
            patch(
                "pawn_diarize.core.queue_listener._resolve_db_dsn",
                return_value="postgresql+psycopg://dummy/dummy",
            ),
            patch("pawn_diarize.core.database.get_engine", return_value=engine),
            patch("pawn_diarize.core.database.init_db"),
            patch(
                "pawn_diarize.core.database.load_session_state",
                return_value=({}, 30.0, ["s3://b/chunk-3.flac"], 10),
            ),
            patch(
                "pawn_diarize.core.session_captures.ingest_session_captures"
            ) as ingest,
            patch(
                "pawn_diarize.core.queue_listener._resolve_audio_paths"
            ) as resolve,
            patch(
                "pawn_diarize.core.combined.transcribe_with_diarization"
            ) as transcribe,
            patch(
                "pawn_diarize.core.queue_listener._post_session_side_effects"
            ) as side,
        ):
            _run_transcribe_diarize(params, cfg)

        resolve.assert_not_called()
        transcribe.assert_not_called()
        ingest.assert_called_once()
        side.assert_called_once()


class TestAlreadySessionCompletedHelper:
    def test_marker_match(self):
        from pawn_server.core.queue_listener import _already_session_completed

        cfg = SimpleNamespace(db_dsn="postgresql+psycopg://dummy/dummy")
        row = MagicMock(
            status="completed",
            prompt="segments=5",
            response="segments=5 items=1",
            started_at=None,
            created_at=datetime.now(timezone.utc),
        )
        fake_db = MagicMock()
        fake_db.__enter__ = MagicMock(return_value=fake_db)
        fake_db.__exit__ = MagicMock(return_value=False)
        fake_db.scalars.return_value.all.return_value = [row]

        with (
            patch("pawn_core.database.get_engine"),
            patch("sqlalchemy.orm.Session", return_value=fake_db),
        ):
            assert _already_session_completed(cfg, "s1", 5) is True
            assert _already_session_completed(cfg, "s1", 6) is False

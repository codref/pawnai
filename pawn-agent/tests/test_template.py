"""Tests for the template expression resolver."""

from __future__ import annotations

import pytest

from pawn_agent.skills.template import (
    TemplateResolutionError,
    resolve_params,
    resolve_value,
)

CTX = {
    "input": {"audio_path": "s3://bucket/audio.wav", "timestamps": True},
    "steps": {
        1: {"local_path": "/tmp/audio.wav", "temp_dir": "/tmp/pawn_123"},
        2: {"transcript": "Hello world.", "session_id": "sess-abc"},
    },
    "context": {"session_id": "ctx-001"},
}


class TestResolveValue:
    def test_plain_string_passthrough(self):
        assert resolve_value("no templates here", CTX) == "no templates here"

    def test_non_string_passthrough(self):
        assert resolve_value(42, CTX) == 42
        assert resolve_value(True, CTX) is True
        assert resolve_value(None, CTX) is None
        assert resolve_value([1, 2], CTX) == [1, 2]

    def test_pure_input_expression(self):
        result = resolve_value("{{input.audio_path}}", CTX)
        assert result == "s3://bucket/audio.wav"

    def test_pure_expression_preserves_type(self):
        # timestamps is bool; pure expression should return bool, not str
        result = resolve_value("{{input.timestamps}}", CTX)
        assert result is True

    def test_step_output_expression(self):
        assert resolve_value("{{steps.1.local_path}}", CTX) == "/tmp/audio.wav"
        assert resolve_value("{{steps.2.transcript}}", CTX) == "Hello world."

    def test_context_expression(self):
        assert resolve_value("{{context.session_id}}", CTX) == "ctx-001"

    def test_embedded_expression_in_string(self):
        result = resolve_value("prefix-{{input.audio_path}}-suffix", CTX)
        assert result == "prefix-s3://bucket/audio.wav-suffix"

    def test_multiple_embedded_expressions(self):
        result = resolve_value("{{steps.2.session_id}}/{{steps.2.transcript}}", CTX)
        assert result == "sess-abc/Hello world."

    def test_missing_key_raises(self):
        with pytest.raises(TemplateResolutionError, match="missing_key"):
            resolve_value("{{input.missing_key}}", CTX)

    def test_missing_step_raises(self):
        with pytest.raises(TemplateResolutionError):
            resolve_value("{{steps.99.transcript}}", CTX)

    def test_missing_nested_key_raises(self):
        with pytest.raises(TemplateResolutionError):
            resolve_value("{{steps.1.nonexistent}}", CTX)


class TestResolveParams:
    def test_resolves_all_values(self):
        params = {
            "audio_path": "{{input.audio_path}}",
            "local_path": "{{steps.1.local_path}}",
            "static_val": "unchanged",
        }
        result = resolve_params(params, CTX)
        assert result["audio_path"] == "s3://bucket/audio.wav"
        assert result["local_path"] == "/tmp/audio.wav"
        assert result["static_val"] == "unchanged"

    def test_null_value_passthrough(self):
        params = {"temp_dir": None}
        result = resolve_params(params, CTX)
        assert result["temp_dir"] is None

    def test_empty_params(self):
        assert resolve_params({}, CTX) == {}

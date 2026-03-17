"""Tests for per-request scanning control via headers (Issue #22)."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock

from secretgate.proxy import (
    _extract_scanning_overrides,
    _remove_secretgate_headers,
    HEADER_MODE,
    HEADER_SKIP,
)
from secretgate.steps import SecretRedactionStep
from secretgate.pipeline import PipelineContext
from secretgate.secrets.scanner import SecretScanner


# ---------------------------------------------------------------------------
# Tests: Header extraction
# ---------------------------------------------------------------------------


class TestExtractScanningOverrides:
    def test_no_headers_returns_defaults(self):
        mode, skip = _extract_scanning_overrides({})
        assert mode is None
        assert skip is False

    def test_mode_audit(self):
        headers = {"X-Secretgate-Mode": "audit"}
        mode, skip = _extract_scanning_overrides(headers)
        assert mode == "audit"
        assert skip is False

    def test_mode_redact(self):
        headers = {"X-Secretgate-Mode": "redact"}
        mode, skip = _extract_scanning_overrides(headers)
        assert mode == "redact"
        assert skip is False

    def test_mode_block(self):
        headers = {"X-Secretgate-Mode": "block"}
        mode, skip = _extract_scanning_overrides(headers)
        assert mode == "block"
        assert skip is False

    def test_mode_case_insensitive(self):
        headers = {"x-secretgate-mode": "AUDIT"}
        mode, skip = _extract_scanning_overrides(headers)
        assert mode == "audit"

    def test_invalid_mode_ignored(self):
        headers = {"X-Secretgate-Mode": "invalid"}
        mode, skip = _extract_scanning_overrides(headers)
        assert mode is None

    def test_skip_true(self):
        headers = {"X-Secretgate-Skip": "true"}
        mode, skip = _extract_scanning_overrides(headers)
        assert skip is True

    def test_skip_1(self):
        headers = {"X-Secretgate-Skip": "1"}
        mode, skip = _extract_scanning_overrides(headers)
        assert skip is True

    def test_skip_yes(self):
        headers = {"X-Secretgate-Skip": "yes"}
        mode, skip = _extract_scanning_overrides(headers)
        assert skip is True

    def test_skip_false(self):
        headers = {"X-Secretgate-Skip": "false"}
        mode, skip = _extract_scanning_overrides(headers)
        assert skip is False

    def test_skip_case_insensitive(self):
        headers = {"x-secretgate-skip": "TRUE"}
        mode, skip = _extract_scanning_overrides(headers)
        assert skip is True

    def test_both_headers(self):
        headers = {
            "X-Secretgate-Mode": "audit",
            "X-Secretgate-Skip": "true",
        }
        mode, skip = _extract_scanning_overrides(headers)
        assert mode == "audit"
        assert skip is True


class TestRemoveSecretgateHeaders:
    def test_removes_secretgate_headers(self):
        headers = {
            "X-Secretgate-Mode": "audit",
            "X-Secretgate-Skip": "true",
            "Authorization": "Bearer token",
            "Content-Type": "application/json",
        }
        result = _remove_secretgate_headers(headers)
        assert "X-Secretgate-Mode" not in result
        assert "X-Secretgate-Skip" not in result
        assert result["Authorization"] == "Bearer token"
        assert result["Content-Type"] == "application/json"

    def test_case_insensitive_removal(self):
        headers = {
            "x-secretgate-mode": "audit",
            "X-SECRETGATE-SKIP": "true",
            "Authorization": "Bearer token",
        }
        result = _remove_secretgate_headers(headers)
        assert len(result) == 1
        assert result["Authorization"] == "Bearer token"

    def test_removes_any_secretgate_header(self):
        headers = {
            "X-Secretgate-Custom": "value",
            "x-secretgate-patterns": "aws,github",
        }
        result = _remove_secretgate_headers(headers)
        assert len(result) == 0

    def test_empty_headers(self):
        result = _remove_secretgate_headers({})
        assert result == {}


# ---------------------------------------------------------------------------
# Tests: Mode override in SecretRedactionStep
# ---------------------------------------------------------------------------


class TestSecretRedactionStepModeOverride:
    @pytest.fixture
    def scanner(self):
        return SecretScanner()

    @pytest.mark.asyncio
    async def test_default_mode_used_without_override(self, scanner):
        step = SecretRedactionStep(scanner, mode="redact")
        ctx = PipelineContext()
        # No mode_override in metadata
        body = {"messages": [{"role": "user", "content": "test AKIAIOSFODNN7EXAMPLE"}]}
        result = await step.process_request(body, ctx)
        # Should use default mode (redact) - result should be modified
        assert result is not None
        # Secrets should be redacted
        assert "REDACTED" in str(result) or result["messages"][0]["content"] != "test AKIAIOSFODNN7EXAMPLE"

    @pytest.mark.asyncio
    async def test_mode_override_to_audit(self, scanner):
        step = SecretRedactionStep(scanner, mode="redact")  # default is redact
        ctx = PipelineContext()
        ctx.metadata["mode_override"] = "audit"
        body = {"messages": [{"role": "user", "content": "test AKIAIOSFODNN7EXAMPLE"}]}
        result = await step.process_request(body, ctx)
        # In audit mode, secrets are logged but not redacted
        assert result is not None
        # Content should be unchanged
        assert result["messages"][0]["content"] == "test AKIAIOSFODNN7EXAMPLE"

    @pytest.mark.asyncio
    async def test_mode_override_to_block(self, scanner):
        step = SecretRedactionStep(scanner, mode="redact")  # default is redact
        ctx = PipelineContext()
        ctx.metadata["mode_override"] = "block"
        body = {"messages": [{"role": "user", "content": "test AKIAIOSFODNN7EXAMPLE"}]}
        result = await step.process_request(body, ctx)
        # In block mode, result should be None
        assert result is None

    @pytest.mark.asyncio
    async def test_mode_override_to_redact_from_audit(self, scanner):
        step = SecretRedactionStep(scanner, mode="audit")  # default is audit
        ctx = PipelineContext()
        ctx.metadata["mode_override"] = "redact"
        body = {"messages": [{"role": "user", "content": "test AKIAIOSFODNN7EXAMPLE"}]}
        result = await step.process_request(body, ctx)
        # In redact mode, secrets should be replaced
        assert result is not None
        assert "REDACTED" in str(result) or result["messages"][0]["content"] != "test AKIAIOSFODNN7EXAMPLE"

    @pytest.mark.asyncio
    async def test_no_secrets_returns_body_unchanged(self, scanner):
        step = SecretRedactionStep(scanner, mode="redact")
        ctx = PipelineContext()
        ctx.metadata["mode_override"] = "block"  # Would block if secrets found
        body = {"messages": [{"role": "user", "content": "just normal text"}]}
        result = await step.process_request(body, ctx)
        # No secrets, so body passes through regardless of mode
        assert result == body

    @pytest.mark.asyncio
    async def test_none_override_uses_default(self, scanner):
        step = SecretRedactionStep(scanner, mode="block")
        ctx = PipelineContext()
        ctx.metadata["mode_override"] = None  # Explicitly None
        body = {"messages": [{"role": "user", "content": "test AKIAIOSFODNN7EXAMPLE"}]}
        result = await step.process_request(body, ctx)
        # Should use default mode (block)
        assert result is None


# ---------------------------------------------------------------------------
# Tests: Header constant values
# ---------------------------------------------------------------------------


class TestHeaderConstants:
    def test_header_mode_value(self):
        assert HEADER_MODE == "x-secretgate-mode"

    def test_header_skip_value(self):
        assert HEADER_SKIP == "x-secretgate-skip"

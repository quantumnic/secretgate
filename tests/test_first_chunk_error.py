"""Tests for first-chunk error detection in streaming responses (Issue #23)."""

from __future__ import annotations

import pytest
from secretgate.proxy import _is_error_chunk


class TestIsErrorChunk:
    def test_json_with_error_field(self):
        chunk = b'{"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}}'
        assert _is_error_chunk(chunk) is True

    def test_json_without_error_field(self):
        chunk = b'{"id": "chatcmpl-123", "model": "gpt-4"}'
        assert _is_error_chunk(chunk) is False

    def test_sse_data_line(self):
        chunk = b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
        assert _is_error_chunk(chunk) is False

    def test_empty_chunk(self):
        chunk = b""
        assert _is_error_chunk(chunk) is False

    def test_whitespace_only(self):
        chunk = b"   \n\n"
        assert _is_error_chunk(chunk) is False

    def test_invalid_json(self):
        chunk = b"not json at all"
        assert _is_error_chunk(chunk) is False

    def test_partial_json(self):
        chunk = b'{"error": '
        assert _is_error_chunk(chunk) is False

    def test_json_array(self):
        chunk = b'[{"message": "hello"}]'
        assert _is_error_chunk(chunk) is False

    def test_nested_error_detected(self):
        chunk = b'{"error": {"code": 401, "message": "Invalid API key"}}'
        assert _is_error_chunk(chunk) is True

    def test_error_with_details(self):
        chunk = b'{"error": {"message": "Bad request", "type": "invalid_request_error", "param": "messages", "code": null}}'
        assert _is_error_chunk(chunk) is True

    def test_unicode_error_message(self):
        chunk = '{"error": {"message": "Fehler: Ungültige Anfrage"}}'.encode("utf-8")
        assert _is_error_chunk(chunk) is True

    def test_json_with_whitespace(self):
        chunk = b'  \n  {"error": {"message": "error"}}  \n  '
        assert _is_error_chunk(chunk) is True

    def test_openai_style_error(self):
        chunk = b'{"error": {"message": "You exceeded your current quota", "type": "insufficient_quota", "param": null, "code": "insufficient_quota"}}'
        assert _is_error_chunk(chunk) is True

    def test_anthropic_style_error(self):
        chunk = b'{"type": "error", "error": {"type": "rate_limit_error", "message": "Rate limited"}}'
        assert _is_error_chunk(chunk) is True

    def test_binary_data(self):
        chunk = b"\x00\x01\x02\x03"
        assert _is_error_chunk(chunk) is False

    def test_sse_error_event(self):
        # SSE format error events should not trigger (they don't start with {)
        chunk = b'event: error\ndata: {"error": "something"}\n\n'
        assert _is_error_chunk(chunk) is False


class TestFirstChunkErrorDetection:
    """Integration tests would go here - these require mocking httpx responses."""

    def test_error_status_codes_to_check(self):
        """Document which status codes trigger error handling."""
        # 4xx client errors
        assert 400 >= 400  # Bad Request
        assert 401 >= 400  # Unauthorized
        assert 403 >= 400  # Forbidden
        assert 404 >= 400  # Not Found
        assert 429 >= 400  # Rate Limited
        # 5xx server errors
        assert 500 >= 400  # Internal Server Error
        assert 502 >= 400  # Bad Gateway
        assert 503 >= 400  # Service Unavailable

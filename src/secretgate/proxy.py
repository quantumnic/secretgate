"""Reverse proxy core — the heart of secretgate.

For each registered provider, we mount routes that:
1. Accept the request from the AI coding tool
2. Run it through the pipeline (secret scanning, etc.)
3. Forward to the real provider API
4. Stream the response back through the pipeline
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, AsyncIterator

import httpx
import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from secretgate.config import ProviderConfig
from secretgate.forward import _AUTH_PATH_PATTERNS
from secretgate.pipeline import Pipeline, PipelineContext

if TYPE_CHECKING:
    from secretgate.server import AppState

logger = structlog.get_logger()

# Header names for per-request scanning control
HEADER_MODE = "x-secretgate-mode"  # audit, redact, block
HEADER_SKIP = "x-secretgate-skip"  # true/false — skip scanning entirely


def _extract_scanning_overrides(headers: dict) -> tuple[str | None, bool]:
    """Extract per-request scanning overrides from headers.

    Returns:
        (mode_override, skip_scanning)
        - mode_override: None or one of "audit", "redact", "block"
        - skip_scanning: True if scanning should be skipped entirely
    """
    mode_override = None
    skip_scanning = False

    # Normalize header keys to lowercase for case-insensitive matching
    headers_lower = {k.lower(): v for k, v in headers.items()}

    # Check for mode override
    if mode_val := headers_lower.get(HEADER_MODE):
        mode_val = mode_val.lower().strip()
        if mode_val in ("audit", "redact", "block"):
            mode_override = mode_val
            logger.debug("per_request_mode_override", mode=mode_override)

    # Check for skip header
    if skip_val := headers_lower.get(HEADER_SKIP):
        skip_val = skip_val.lower().strip()
        if skip_val in ("true", "1", "yes"):
            skip_scanning = True
            logger.debug("per_request_skip_scanning")

    return mode_override, skip_scanning


def _remove_secretgate_headers(headers: dict) -> dict:
    """Remove secretgate-specific headers before forwarding to upstream."""
    return {
        k: v for k, v in headers.items()
        if not k.lower().startswith("x-secretgate-")
    }


def create_provider_router(
    provider: ProviderConfig,
    pipeline: Pipeline,
    state: AppState,
) -> APIRouter:
    """Create a FastAPI router that proxies all requests for a provider."""
    router = APIRouter(prefix=f"/{provider.name}")

    @router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def proxy(request: Request, path: str):
        upstream_url = f"{provider.base_url.rstrip('/')}/{path}"
        if request.url.query:
            upstream_url = f"{upstream_url}?{request.url.query}"

        # Forward headers, replacing host
        headers = dict(request.headers)
        headers.pop("host", None)
        headers.pop("content-length", None)

        # Extract per-request scanning overrides from headers
        mode_override, skip_scanning = _extract_scanning_overrides(headers)

        # Remove secretgate headers before forwarding to upstream
        headers = _remove_secretgate_headers(headers)

        # For non-JSON, GET, or auth/token endpoints, pass through directly
        if (
            request.method == "GET"
            or "application/json" not in request.headers.get("content-type", "")
            or _AUTH_PATH_PATTERNS.search(f"/{path}")
        ):
            return await _passthrough(request, upstream_url, headers, state.http_client)

        # If skip_scanning is requested, pass through without pipeline
        if skip_scanning:
            logger.info("scanning_skipped_per_request", path=path)
            raw_body = await request.body()
            return await _forward_raw(raw_body, upstream_url, headers, state.http_client)

        # Parse JSON body
        raw_body = await request.body()
        try:
            body = json.loads(raw_body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return await _forward_raw(raw_body, upstream_url, headers, state.http_client)

        # Run request pipeline with optional mode override
        ctx = PipelineContext()
        ctx.metadata["mode_override"] = mode_override  # Pass to pipeline steps
        result = await pipeline.run_request(body, ctx)

        if result is None:
            # Pipeline blocked the request
            return JSONResponse(
                status_code=403,
                content={
                    "error": {
                        "message": f"Request blocked by secretgate: {len(ctx.alerts)} secret(s) detected",
                        "type": "security_block",
                        "alerts": ctx.alerts,
                    }
                },
            )

        is_streaming = result.get("stream", False)
        modified_body = json.dumps(result).encode("utf-8")
        headers["content-length"] = str(len(modified_body))

        if is_streaming:
            return await _forward_streaming(
                modified_body, upstream_url, headers, state.http_client, pipeline, ctx
            )
        else:
            return await _forward_buffered(
                modified_body, upstream_url, headers, state.http_client, pipeline, ctx
            )

    return router


async def _passthrough(
    request: Request, url: str, headers: dict, client: httpx.AsyncClient
) -> StreamingResponse:
    """Pass non-JSON requests through without processing."""
    resp = await client.request(
        method=request.method,
        url=url,
        headers=headers,
        content=await request.body(),
    )
    return StreamingResponse(
        content=iter([resp.content]),
        status_code=resp.status_code,
        headers=dict(resp.headers),
    )


async def _forward_raw(
    body: bytes, url: str, headers: dict, client: httpx.AsyncClient
) -> StreamingResponse:
    """Forward unparseable body as-is."""
    resp = await client.request(method="POST", url=url, headers=headers, content=body)
    return StreamingResponse(
        content=iter([resp.content]),
        status_code=resp.status_code,
        headers=dict(resp.headers),
    )


async def _forward_buffered(
    body: bytes,
    url: str,
    headers: dict,
    client: httpx.AsyncClient,
    pipeline: Pipeline,
    ctx: PipelineContext,
) -> JSONResponse:
    """Forward request and process complete response through pipeline."""
    resp = await client.request(method="POST", url=url, headers=headers, content=body)

    try:
        resp_body = resp.json()
        processed = await pipeline.run_response(resp_body, ctx)
        return JSONResponse(content=processed, status_code=resp.status_code)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JSONResponse(content=resp.text, status_code=resp.status_code)


async def _forward_streaming(
    body: bytes,
    url: str,
    headers: dict,
    client: httpx.AsyncClient,
    pipeline: Pipeline,
    ctx: PipelineContext,
) -> StreamingResponse | JSONResponse:
    """Forward request and process streaming response chunks through pipeline.

    Implements first-chunk error detection (Issue #23): if the upstream API
    returns an error status code, convert it to a proper HTTP error response
    instead of streaming it back to the client.
    """
    # Use a context manager with stream to get status code before iterating
    async with client.stream("POST", url, headers=headers, content=body) as resp:
        # Check if upstream returned an error status
        if resp.status_code >= 400:
            # Read the error body and return as proper HTTP error
            error_body = await resp.aread()
            logger.warning(
                "streaming_error_detected",
                status_code=resp.status_code,
                url=url,
            )
            try:
                error_json = json.loads(error_body)
                return JSONResponse(status_code=resp.status_code, content=error_json)
            except (json.JSONDecodeError, UnicodeDecodeError):
                return JSONResponse(
                    status_code=resp.status_code,
                    content={"error": {"message": error_body.decode(errors="replace")}}
                )

        # For successful responses, peek at first chunk to detect SSE errors
        first_chunk = None
        async for chunk in resp.aiter_bytes():
            first_chunk = chunk
            break

        # If no chunks at all, return empty response
        if first_chunk is None:
            return StreamingResponse(content=iter([]), media_type="text/event-stream")

        # Check if first chunk looks like an error response (JSON with error field)
        # Some APIs return 200 OK with error in body for streaming
        if _is_error_chunk(first_chunk):
            logger.warning(
                "streaming_error_in_body",
                url=url,
            )
            try:
                error_json = json.loads(first_chunk)
                # Extract status code from error if available
                error_status = error_json.get("error", {}).get("status", 500)
                if isinstance(error_status, str):
                    error_status = 500
                return JSONResponse(status_code=error_status, content=error_json)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass  # Not JSON, continue with normal streaming

        # Normal streaming: yield first chunk, then rest
        async def stream_with_first_chunk() -> AsyncIterator[bytes]:
            processed = await pipeline.run_response_chunk(first_chunk, ctx)
            yield processed
            async for chunk in resp.aiter_bytes():
                processed = await pipeline.run_response_chunk(chunk, ctx)
                yield processed

        return StreamingResponse(
            content=stream_with_first_chunk(),
            media_type="text/event-stream"
        )


def _is_error_chunk(chunk: bytes) -> bool:
    """Check if a chunk looks like an error response.

    Returns True if the chunk is valid JSON containing an "error" field.
    This catches cases where APIs return 200 OK with an error in the body.
    """
    try:
        # Only check if it looks like JSON (starts with '{')
        text = chunk.decode("utf-8").strip()
        if not text.startswith("{"):
            return False
        data = json.loads(text)
        return "error" in data
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False

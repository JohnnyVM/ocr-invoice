"""Provider-neutral, synchronous HTTP extraction client."""

from __future__ import annotations

import json
import logging
import os
import random
import time
import uuid
import warnings
from collections.abc import Callable
from pathlib import Path
from time import sleep
from typing import Any, Protocol, Self, TypeAlias
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)
JsonValue: TypeAlias = Any


class ExtractionRequest(BaseModel):
    document_path: Path
    prompt: str | None = None
    output_schema: dict[str, Any] | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = None

    @field_validator("prompt")
    @classmethod
    def empty_prompt_is_none(cls, value: str | None) -> str | None:
        return value.strip() or None if value is not None else None


class DocumentElement(BaseModel):
    element_type: str
    text: str | None = None
    bounding_box: tuple[float, float, float, float] | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    page: int = Field(default=1, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("element_type")
    @classmethod
    def nonblank_type(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("element_type must not be blank")
        return value

    @field_validator("bounding_box")
    @classmethod
    def valid_box(
        cls, value: tuple[float, float, float, float] | None
    ) -> tuple[float, float, float, float] | None:
        if value is not None and (value[0] > value[2] or value[1] > value[3]):
            raise ValueError("bounding_box must be (x_min, y_min, x_max, y_max)")
        return value


class ExtractionResult(BaseModel):
    text: str = ""
    structured_data: dict[str, Any] = Field(default_factory=dict)
    elements: list[DocumentElement] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractionTelemetry(BaseModel):
    request_id: str
    attempts: int
    elapsed_ms: float
    server_processing_ms: float | None = None
    http_status: int
    endpoint_host: str


class ExtractionResponse(BaseModel):
    result: ExtractionResult
    telemetry: ExtractionTelemetry
    raw_response: JsonValue


class ExtractorClient(Protocol):
    def extract(self, request: ExtractionRequest) -> ExtractionResponse: ...

    def close(self) -> None: ...


class ExtractorClientError(Exception):
    category = "extractor_client"

    def __init__(
        self,
        message: str,
        *,
        request_id: str = "",
        attempts: int = 0,
        retryable: bool = False,
        http_status: int | None = None,
        server_error_code: str | None = None,
    ) -> None:
        self.request_id = request_id
        self.attempts = attempts
        self.retryable = retryable
        self.http_status = http_status
        self.server_error_code = server_error_code
        self.message = _safe_excerpt(message)
        super().__init__(self.message)


class ExtractorConfigurationError(ExtractorClientError):
    category = "configuration"


class ExtractorInputError(ExtractorClientError):
    category = "input"


class ExtractorConnectionError(ExtractorClientError):
    category = "connection"


class ExtractorTimeoutError(ExtractorClientError):
    category = "timeout"


class ExtractorHttpError(ExtractorClientError):
    category = "http"


class ExtractorResponseDecodeError(ExtractorClientError):
    category = "response_decode"


class ExtractorResponseSchemaError(ExtractorClientError):
    category = "response_schema"


class HttpExtractorConfig(BaseModel):
    base_url: str
    extraction_path: str = "/v1/extractions"
    timeout_seconds: float = 300.0
    connect_timeout_seconds: float = 10.0
    max_attempts: int = Field(default=2, ge=1, le=5)
    retry_backoff_seconds: float = 0.5
    bearer_token_env: str | None = None
    verify_tls: bool = True
    user_agent: str = "document-lab/0.1"

    @model_validator(mode="after")
    def validate_config(self) -> HttpExtractorConfig:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("base_url must use http or https and include a host")
        if not self.extraction_path.startswith("/"):
            raise ValueError("extraction_path must start with '/'")
        if self.timeout_seconds <= 0 or self.connect_timeout_seconds <= 0:
            raise ValueError("timeouts must be positive")
        if self.retry_backoff_seconds <= 0:
            raise ValueError("retry_backoff_seconds must be positive")
        if parsed.scheme == "http" and not _is_loopback(parsed.hostname):
            warnings.warn("Using plain HTTP for a non-loopback endpoint", UserWarning)
        if not self.verify_tls:
            warnings.warn("TLS certificate verification is disabled", UserWarning)
        return self


class ExtractorConfig(BaseModel):
    type: str
    http: HttpExtractorConfig | None = None

    @model_validator(mode="after")
    def validate_type(self) -> ExtractorConfig:
        if self.type == "http" and self.http is None:
            raise ValueError("http configuration is required for type 'http'")
        return self


def _safe_excerpt(
    value: str, limit: int = 2048, secrets: tuple[str, ...] = ()
) -> str:
    for secret in secrets:
        if secret:
            value = value.replace(secret, "[REDACTED]")
    return " ".join(value.replace("\x00", "").split())[:limit]


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress_is_loopback(host)
    except ValueError:
        return False


def ipaddress_is_loopback(host: str) -> bool:
    import ipaddress

    return ipaddress.ip_address(host).is_loopback


def _mime_from_signature(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            signature = handle.read(8)
    except OSError as exc:
        raise ExtractorInputError(f"Cannot read document: {exc}") from exc
    if signature.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if signature == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    raise ExtractorInputError("Document is not a supported JPEG or PNG image")


class HttpExtractorClient:
    """HTTP implementation of :class:`ExtractorClient`."""

    def __init__(
        self,
        config: HttpExtractorConfig,
        *,
        client: httpx.Client | None = None,
        sleep_fn: Callable[[float], None] = sleep,
        random_fn: Callable[[], float] = random.random,
    ) -> None:
        self.config = config
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(
                config.timeout_seconds, connect=config.connect_timeout_seconds
            ),
            verify=config.verify_tls,
        )
        self._sleep = sleep_fn
        self._random = random_fn
        self._closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._client.close()
            self._closed = True

    def extract(self, request: ExtractionRequest) -> ExtractionResponse:
        request_id = request.request_id or str(uuid.uuid4())
        if self._closed:
            raise ExtractorClientError("Extractor client is closed", request_id=request_id)
        path = request.document_path
        if not path.exists() or not path.is_file():
            raise ExtractorInputError("Document path must be an existing regular file", request_id=request_id)
        try:
            if path.stat().st_size == 0:
                raise ExtractorInputError("Document is empty", request_id=request_id)
        except OSError as exc:
            raise ExtractorInputError("Document cannot be inspected", request_id=request_id) from exc
        mime = _mime_from_signature(path)
        token = self._token(request_id)
        payload = {
            "schema_version": "1",
            "request_id": request_id,
            "prompt": request.prompt,
            "output_schema": request.output_schema,
            "options": request.options,
        }
        started = time.perf_counter()
        attempts = 0
        last_error: ExtractorClientError | None = None
        endpoint = self.config.base_url.rstrip("/") + self.config.extraction_path
        for attempt in range(1, self.config.max_attempts + 1):
            attempts = attempt
            response: httpx.Response | None = None
            try:
                with path.open("rb") as document:
                    response = self._client.post(
                        endpoint,
                        files={"document": (path.name, document, mime)},
                        data={"request": json.dumps(payload, separators=(",", ":"))},
                        headers=self._headers(request_id, token),
                    )
            except httpx.TimeoutException:
                last_error = ExtractorTimeoutError("Extractor request timed out", request_id=request_id, attempts=attempt, retryable=True)
            except httpx.RequestError:
                last_error = ExtractorConnectionError("Extractor connection failed", request_id=request_id, attempts=attempt, retryable=True)
            else:
                if response.is_success:
                    return self._success(response, request_id, attempts, started)
                last_error = self._http_error(response, request_id, attempt, token)

            if not last_error.retryable or attempt == self.config.max_attempts:
                break
            delay = self._retry_delay(last_error, response, attempt)
            logger.debug("Retrying extraction request_id=%s delay=%.3f", request_id, delay)
            self._sleep(delay)
        assert last_error is not None
        elapsed = (time.perf_counter() - started) * 1000
        logger.info("Extraction request_id=%s host=%s attempts=%d status=%s elapsed_ms=%.1f", request_id, self._host(), attempts, last_error.category, elapsed)
        raise last_error

    def _headers(self, request_id: str, token: str | None) -> dict[str, str]:
        headers = {"Accept": "application/json", "User-Agent": self.config.user_agent, "X-Request-ID": request_id}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _token(self, request_id: str) -> str | None:
        if not self.config.bearer_token_env:
            return None
        token = os.environ.get(self.config.bearer_token_env, "").strip()
        if not token:
            raise ExtractorConfigurationError("Configured bearer token is unavailable", request_id=request_id)
        return token

    def _host(self) -> str:
        return urlparse(self.config.base_url).hostname or "unknown"

    def _success(self, response: httpx.Response, request_id: str, attempts: int, started: float) -> ExtractionResponse:
        try:
            raw = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise ExtractorResponseDecodeError("Successful response is not valid JSON", request_id=request_id, attempts=attempts) from exc
        try:
            if not isinstance(raw, dict):
                raise TypeError("successful response must be a JSON object")
            if raw.get("schema_version") != "1":
                raise ValueError("unsupported schema_version")
            if raw.get("request_id") != request_id:
                raise ValueError("response request_id does not match request")
            processing = raw.get("processing") or {}
            if not isinstance(processing, dict):
                raise TypeError("processing must be an object")
            duration = processing.get("duration_ms")
            if duration is not None and (not isinstance(duration, (int, float)) or duration < 0):
                raise ValueError("processing.duration_ms must be non-negative")
            result_payload = raw.get("result")
            if not isinstance(result_payload, dict):
                raise TypeError("result must be an object")
            result = ExtractionResult.model_validate(result_payload)
        except (ValueError, TypeError) as exc:
            raise ExtractorResponseSchemaError(str(exc), request_id=request_id, attempts=attempts) from exc
        elapsed = (time.perf_counter() - started) * 1000
        telemetry = ExtractionTelemetry(request_id=request_id, attempts=attempts, elapsed_ms=elapsed, server_processing_ms=duration, http_status=response.status_code, endpoint_host=self._host())
        logger.info("Extraction request_id=%s host=%s attempts=%d status=success elapsed_ms=%.1f", request_id, self._host(), attempts, elapsed)
        return ExtractionResponse(result=result, telemetry=telemetry, raw_response=raw)

    def _http_error(
        self,
        response: httpx.Response,
        request_id: str,
        attempt: int,
        token: str | None,
    ) -> ExtractorHttpError:
        try:
            raw = response.json()
        except ValueError:
            raw = None
        error = raw.get("error", {}) if isinstance(raw, dict) else {}
        retryable = bool(error.get("retryable", False)) if isinstance(error, dict) else False
        code = error.get("code") if isinstance(error, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
        body = message or _safe_excerpt(response.text, secrets=(token,) if token else ())
        body = _safe_excerpt(str(body), secrets=(token,) if token else ())
        retryable = retryable or response.status_code in {408, 429, 502, 503, 504}
        return ExtractorHttpError(body or f"Extractor returned HTTP {response.status_code}", request_id=request_id, attempts=attempt, retryable=retryable, http_status=response.status_code, server_error_code=code)

    def _retry_delay(self, error: ExtractorClientError, response: httpx.Response | None, attempt: int) -> float:
        if response is not None and response.status_code in {429, 503}:
            try:
                return min(30.0, max(0.0, float(response.headers.get("Retry-After", ""))))
            except ValueError:
                pass
        return self.config.retry_backoff_seconds * (2 ** (attempt - 1)) + self._random() * 0.1


def create_extractor(config: ExtractorConfig) -> ExtractorClient:
    if config.type == "http" and config.http is not None:
        return HttpExtractorClient(config.http)
    raise ExtractorConfigurationError(f"Unsupported extractor type: {config.type}")

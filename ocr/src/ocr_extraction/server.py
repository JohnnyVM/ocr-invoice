"""HTTP inference server for the local Qwen3-VL Ollama runtime."""

from __future__ import annotations

import base64
import json
import os
import time
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from .extractor import ExtractionResult

SUPPORTED_SCHEMA_VERSION = "1"
DEFAULT_MODEL = "qwen3-vl:4b-instruct"


def _error(
    code: str,
    message: str,
    request_id: str,
    *,
    retryable: bool = False,
    status_code: int = 400,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "retryable": retryable,
                "details": {},
            },
            "request_id": request_id,
        },
    )


def _image_mime(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    return None


def _model_prompt(request: dict[str, Any]) -> str:
    schema = request.get("output_schema")
    schema_instruction = (
        "The structured_data object must follow this JSON Schema:\n"
        + json.dumps(schema, ensure_ascii=False)
        if schema
        else "Return the fields you can reliably read in structured_data."
    )
    prompt = request.get("prompt") or "Extract the document faithfully."
    return (
        "You are a document extraction model. Return only one JSON object with "
        "the keys text, structured_data, elements, and metadata. "
        "Use an empty array for elements when no layout coordinates are available. "
        "Do not invent values.\n\n"
        f"Task: {prompt}\n{schema_instruction}"
    )


def _normalize_model_output(raw_text: str, model: str) -> ExtractionResult:
    metadata = {"provider": "ollama", "model": model}
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return ExtractionResult(text=raw_text, metadata={**metadata, "parse_error": True})
    if not isinstance(parsed, dict):
        return ExtractionResult(
            text=raw_text,
            metadata={**metadata, "parse_error": "JSON result was not an object"},
        )
    if any(key in parsed for key in ("text", "structured_data", "elements", "metadata")):
        result = ExtractionResult.model_validate(parsed)
        result.metadata = {**metadata, **result.metadata}
        return result
    return ExtractionResult(structured_data=parsed, metadata=metadata)


class OllamaVisionServer:
    def __init__(
        self,
        *,
        ollama_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.ollama_url = (
            ollama_url or os.getenv("OLLAMA_URL") or "http://127.0.0.1:11434"
        ).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL") or DEFAULT_MODEL
        self.timeout_seconds = timeout_seconds or float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))
        self.client = client or httpx.Client(timeout=self.timeout_seconds)

    def close(self) -> None:
        self.client.close()

    def extract(self, image: bytes, request: dict[str, Any]) -> tuple[ExtractionResult, float]:
        started = time.perf_counter()
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": _model_prompt(request),
            "images": [base64.b64encode(image).decode("ascii")],
            "stream": False,
            "format": "json",
        }
        response = self.client.post(f"{self.ollama_url}/api/generate", json=payload)
        response.raise_for_status()
        body = response.json()
        raw_text = body.get("response")
        if not isinstance(raw_text, str):
            raise TypeError("Ollama response did not contain text")
        return _normalize_model_output(raw_text, self.model), (time.perf_counter() - started) * 1000


def create_app(server: OllamaVisionServer | None = None) -> FastAPI:
    inference = server or OllamaVisionServer()
    app = FastAPI(title="Document extractor", version="1")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": inference.model}

    @app.post("/v1/extractions")
    def extract(document: UploadFile = File(...), request: str = Form(...)) -> JSONResponse:  # noqa: B008
        request_id = ""
        try:
            metadata = json.loads(request)
            if not isinstance(metadata, dict):
                raise TypeError("request must be a JSON object")
            request_id = str(metadata.get("request_id", ""))
            if metadata.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
                return _error("unsupported_schema", "Unsupported request schema version", request_id)
            if not request_id:
                return _error("invalid_request", "request_id is required", request_id)
            image = document.file.read()
            if not image:
                return _error("empty_document", "The uploaded document is empty", request_id)
            if _image_mime(image) is None:
                return _error("unsupported_document", "Only JPEG and PNG documents are supported", request_id)
            result, duration_ms = inference.extract(image, metadata)
        except json.JSONDecodeError:
            return _error("invalid_request", "request is not valid JSON", request_id)
        except httpx.HTTPStatusError as exc:
            return _error("inference_unavailable", f"Inference service returned HTTP {exc.response.status_code}", request_id, retryable=True, status_code=502)
        except (httpx.RequestError, TypeError, ValueError) as exc:
            return _error("inference_unavailable", str(exc), request_id, retryable=True, status_code=502)
        return JSONResponse(
            content={
                "schema_version": SUPPORTED_SCHEMA_VERSION,
                "request_id": request_id,
                "result": result.model_dump(mode="json"),
                "processing": {"duration_ms": duration_ms},
            }
        )

    return app


app = create_app()

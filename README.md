# Document Lab

Document Lab is a local Python laboratory for reproducible document-image extraction experiments. It keeps preprocessing, validation, evaluation, measurement, and tracking independent from the inference provider.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/) are recommended:

```bash
uv sync --extra dev
```

The pipeline API is synchronous because one document experiment is naturally a bounded command-line or notebook operation. HTTP and Ollama calls use explicit timeouts.

## Run an experiment

```bash
uv run document-lab extract image.png --config configs/ollama_qwen.yaml
uv run document-lab benchmark data/manifest.example.jsonl --config configs/fake.yaml
uv run document-lab validate runs/<run>/result.json
```

Use `configs/http_local.yaml` for an inference service implementing `POST /v1/extractions` with a multipart `file` field. A bearer token can be selected with `bearer_token_env`; its value is never written to artifacts.

Every document run writes a unique local artifact directory containing the resolved configuration, preprocessed image, normalized extraction, validation findings, result JSON, and a Markdown report. Original images are not copied by default. The fake adapter is intentionally deterministic for local smoke tests.

## Odoo read-only dataset export

With an approved `.env` containing `RUN_ODOO_INTEGRATION=1`, `ODOO_URL`, `ODOO_DB`, `ODOO_USER`, and `ODOO_API_KEY`, run:

```bash
python scripts/download_odoo_ground_truth.py
```

The exporter uses only Odoo authentication plus `search_read` calls against `ir.attachment`, `stock.picking`, and `stock.move`. It downloads existing PDF attachments, converts the first PDF page locally to JPEG, writes movement metadata to `data/ground_truth`, and creates `data/manifest.jsonl`. It never creates, updates, deletes, validates, confirms, or renders an Odoo record. Credentials are not written to generated files.

The current server snapshot contained 3 existing PDFs linked to `stock.picking` and enough PDFs linked to `sale.order` to complete the requested 20. The exporter prefers stock-picking PDFs and fills the remainder with sale-order PDFs. It intentionally does not substitute invoices, payments, or unrelated `mail.message` attachments.

## MLflow and Jupyter

Tracking is optional. Set `tracking.enabled: true` and optionally `tracking.uri`, then start a local server separately if desired:

```bash
uv run mlflow server --host 127.0.0.1 --port 5000
uv run jupyter lab
```

Each document uses one MLflow run. Tracking failures are retained as warnings unless `tracking.strict` is enabled.

The reusable OCR code is packaged under `ocr/src/ocr_extraction`. The Docker
Compose service mounts `ocr/` at `/home/jovyan/ocr`, so a notebook can install
the live source without rebuilding the image:

```python
%pip install --no-deps --no-build-isolation -e /home/jovyan/ocr
```

Start the configured Jupyter Lab environment from the repository root:

```bash
docker compose -f docker/jupyter/compose.yaml up --build
```

## Verification

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest --cov=document_lab
```

## Design decisions and limitations

Text evaluation uses NFC Unicode normalization, normalized line endings, collapsed whitespace, and trimming. String fields are case-folded with collapsed whitespace; SKU/EAN fields compare digits; money fields use `Decimal`. Resource sampling polls process RSS every 100 ms and reports device-wide GPU memory when NVML is available. p95 benchmark aggregation is reserved for a future benchmark-summary refinement; individual run timings are currently reported.

The Ollama adapter expects `/api/generate` and sends one base64 image with `stream: false`. Structured JSON is parsed when present; otherwise the raw response text is retained with parse-error metadata. Perspective correction, graphical barcode decoding, PaddleOCR-VL serving, and Odoo integration are deliberately out of scope.

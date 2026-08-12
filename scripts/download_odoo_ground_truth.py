"""Read existing Odoo stock-picking and sale-order PDF attachments into the local dataset.

This script deliberately uses only Odoo authentication, search, and read calls.
It never creates, updates, deletes, confirms, validates, or renders anything in Odoo.

TODO use manifest.jsonl to avoid re-downloading existing attachments, and to allow downloading more than 20 attachments in total.
"""

from __future__ import annotations

import base64
import json
import os
import re
import ssl
import urllib.parse
import urllib.request
import xmlrpc.client
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data" / "samples" / "pdf"
JPEG_DIR = ROOT / "data" / "samples"
TRUTH_DIR = ROOT / "data" / "ground_truth"


def load_env() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", line)
        if match:
            values[match.group(1)] = match.group(2).strip().strip('"').strip("'")
    return values


class OdooReadClient:
    def __init__(self, values: dict[str, str]) -> None:
        self.url = values["ODOO_URL"].rstrip("/")
        self.db = values["ODOO_DB"]
        self.user = values["ODOO_USER"]
        self.api_key = values["ODOO_API_KEY"]
        common = xmlrpc.client.ServerProxy(f"{self.url}/xmlrpc/2/common", context=ssl.create_default_context())
        self.uid = common.authenticate(self.db, self.user, self.api_key, {})
        if not self.uid:
            raise RuntimeError("Odoo authentication failed")
        self.models = xmlrpc.client.ServerProxy(f"{self.url}/xmlrpc/2/object", context=ssl.create_default_context())

    def search_read(self, model: str, domain: list[Any], fields: list[str], limit: int = 0) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {"fields": fields}
        if limit:
            kwargs["limit"] = limit
        return self.models.execute_kw(self.db, self.uid, self.api_key, model, "search_read", [domain], kwargs)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "document"


def download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "document-lab-read-only/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        target.write_bytes(response.read())


def pdf_to_jpeg(pdf_path: Path, jpeg_path: Path) -> None:
    try:
        import pymupdf as fitz
    except ImportError as exc:
        raise RuntimeError("PDF to JPEG conversion requires PyMuPDF; install it with `uv add pymupdf`") from exc
    document = fitz.open(pdf_path)
    try:
        if not document.page_count:
            raise RuntimeError(f"PDF has no pages: {pdf_path}")
        page = document.load_page(0)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        pixmap.save(jpeg_path)
    finally:
        document.close()


def main() -> None:
    values = load_env()
    if values.get("RUN_ODOO_INTEGRATION", "0").lower() not in {"1", "true", "yes"}:
        raise SystemExit("RUN_ODOO_INTEGRATION is not enabled")
    client = OdooReadClient(values)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    JPEG_DIR.mkdir(parents=True, exist_ok=True)
    TRUTH_DIR.mkdir(parents=True, exist_ok=True)

    attachment_fields = ["id", "name", "res_id", "res_model", "mimetype", "datas", "file_size", "create_date"]
    stock_attachments = client.search_read(
        "ir.attachment",
        [["res_model", "=", "stock.picking"], ["res_id", "!=", False], ["mimetype", "=", "application/pdf"]],
        attachment_fields,
        limit=20,
    )
    sale_attachments = client.search_read(
        "ir.attachment",
        [["res_model", "=", "sale.order"], ["res_id", "!=", False], ["mimetype", "=", "application/pdf"]],
        attachment_fields,
        limit=max(0, 20 - len(stock_attachments)),
    )
    attachments = stock_attachments + sale_attachments
    if len(attachments) < 20:
        print(f"Warning: found {len(attachments)} existing stock-picking/sale-order PDF attachments; 20 were requested")

    picking_ids = [int(item["res_id"]) for item in stock_attachments]
    pickings = client.search_read(
        "stock.picking",
        [["id", "in", picking_ids]],
        ["id", "name", "origin", "state", "date", "scheduled_date", "partner_id", "picking_type_id", "location_id", "location_dest_id", "move_ids", "company_id", "note"],
    )
    by_id = {int(item["id"]): item for item in pickings}
    move_ids = sorted({int(move_id) for picking in pickings for move_id in picking.get("move_ids", [])})
    moves = client.search_read(
        "stock.move",
        [["id", "in", move_ids]],
        ["id", "name", "product_id", "product_uom_qty", "quantity", "product_uom", "price_unit", "origin", "state"],
    ) if move_ids else []
    moves_by_picking: dict[int, list[dict[str, Any]]] = {int(picking["id"]): [] for picking in pickings}
    move_by_id = {int(move["id"]): move for move in moves}
    for picking in pickings:
        picking_move_ids = [int(move_id) for move_id in picking.get("move_ids", [])]
        moves_by_picking[int(picking["id"])] = [move_by_id[move_id] for move_id in picking_move_ids if move_id in move_by_id]

    sale_ids = [int(item["res_id"]) for item in sale_attachments]
    sales = client.search_read(
        "sale.order",
        [["id", "in", sale_ids]],
        ["id", "name", "origin", "state", "date_order", "validity_date", "partner_id", "company_id", "currency_id", "amount_untaxed", "amount_tax", "amount_total", "invoice_status", "order_line", "picking_ids"],
    ) if sale_ids else []
    sales_by_id = {int(item["id"]): item for item in sales}
    sale_line_ids = sorted({int(line_id) for sale in sales for line_id in sale.get("order_line", [])})
    sale_lines = client.search_read(
        "sale.order.line",
        [["id", "in", sale_line_ids]],
        ["id", "name", "product_id", "product_uom_qty", "qty_delivered", "product_uom", "price_unit", "discount", "price_subtotal", "price_total"],
    ) if sale_line_ids else []
    sale_lines_by_id = {int(line["id"]): line for line in sale_lines}

    manifest: list[dict[str, Any]] = []
    for index, attachment in enumerate(attachments, start=1):
        record_id = int(attachment["res_id"])
        model = str(attachment["res_model"])
        record = by_id.get(record_id, {"id": record_id}) if model == "stock.picking" else sales_by_id.get(record_id, {"id": record_id})
        document_id = safe_name(str(record.get("name") or f"{model.replace('.', '-')}-{record_id}"))
        stem = f"{index:02d}-{document_id}"
        pdf_path = PDF_DIR / f"{stem}.pdf"
        jpeg_path = JPEG_DIR / f"{stem}.jpeg"
        truth_path = TRUTH_DIR / f"{stem}.json"
        pdf_path.write_bytes(base64.b64decode(attachment["datas"]))
        pdf_to_jpeg(pdf_path, jpeg_path)
        if model == "stock.picking":
            movement = record
            lines = moves_by_picking.get(record_id, [])
            tags = ["odoo", "stock_picking", "read_only_export"]
        else:
            movement = record
            sale_order_line_ids = [int(line_id) for line_id in record.get("order_line", [])]
            lines = [sale_lines_by_id[line_id] for line_id in sale_order_line_ids if line_id in sale_lines_by_id]
            tags = ["odoo", "sale_order", "read_only_export"]
        truth = {"document_id": document_id, "source": {"model": model, "id": record_id, "attachment_id": attachment["id"]}, "movement": movement, "moves": lines, "pdf_attachment": {"name": attachment["name"], "create_date": attachment.get("create_date"), "file_size": attachment.get("file_size")}}
        truth_path.write_text(json.dumps(truth, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        manifest.append({"document_id": document_id, "image_path": str(jpeg_path.relative_to(ROOT / "data")).replace("\\", "/"), "pdf_path": str(pdf_path.relative_to(ROOT / "data")).replace("\\", "/"), "ground_truth_path": str(truth_path.relative_to(ROOT / "data")).replace("\\", "/"), "tags": tags, "source_model": model, "source_attachment_id": attachment["id"]})

    (ROOT / "data" / "manifest.jsonl").write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in manifest) + "\n", encoding="utf-8")
    print(f"Downloaded {len(manifest)} existing Odoo PDF attachments")
    print("PDF directory:", PDF_DIR)
    print("Ground truth directory:", TRUTH_DIR)
    print("Manifest:", ROOT / "data" / "manifest.jsonl")
    print("Only existing PDF attachments were read; no Odoo records were changed.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from .config import PRODUCTS_DIR, ensure_directories, slugify


def product_directory(product_id: str) -> Path:
    ensure_directories()
    path = PRODUCTS_DIR / product_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_upload(product_id: str, filename: str, content: bytes) -> Path:
    extension = Path(filename).suffix.lower()
    if extension not in {".pdf", ".txt", ".md"}:
        raise ValueError("Only PDF, TXT, and Markdown documents are supported.")
    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", Path(filename).name)
    path = product_directory(product_id) / safe_name
    path.write_bytes(content)
    return path


def _source_map_path(product_id: str) -> Path:
    return product_directory(product_id) / ".source_urls.json"


def source_urls(product_id: str) -> dict[str, str]:
    path = _source_map_path(product_id)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _record_source_url(product_id: str, filename: str, url: str) -> None:
    urls = source_urls(product_id)
    urls[filename] = url
    _source_map_path(product_id).write_text(json.dumps(urls, indent=2), encoding="utf-8")


def fetch_webpage(product_id: str, url: str) -> Path:
    """Download one public support page as a local Markdown source for indexing."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Enter a public http(s) URL.")
    response = requests.get(url, timeout=20, headers={"User-Agent": "ProductSupportAssistant/1.0"})
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    if "pdf" in content_type or parsed.path.lower().endswith(".pdf"):
        filename = f"web_{slugify(parsed.netloc + parsed.path)}.pdf"
        path = product_directory(product_id) / filename
        path.write_bytes(response.content)
    else:
        soup = BeautifulSoup(response.text, "html.parser")
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()
        title = soup.title.get_text(" ", strip=True) if soup.title else parsed.netloc
        body = soup.get_text("\n", strip=True)
        if len(body) < 80:
            raise ValueError("The page did not contain enough readable documentation text.")
        filename = f"web_{slugify(parsed.netloc + parsed.path)}.md"
        path = product_directory(product_id) / filename
        path.write_text(f"# {title}\n\n{body}", encoding="utf-8")
    _record_source_url(product_id, path.name, url)
    return path


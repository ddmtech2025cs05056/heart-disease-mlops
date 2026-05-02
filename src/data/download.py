"""Download the UCI Heart Disease (Cleveland) dataset.

Run with: ``python -m src.data.download``
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import requests

from src.config import COLUMNS, RAW_DIR, RAW_FILENAME, UCI_URL

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def download(url: str = UCI_URL, dest: Path | None = None) -> Path:
    """Download the raw Cleveland CSV from the UCI ML repository.

    Returns the path of the downloaded file. Idempotent.
    """
    dest = dest or (RAW_DIR / RAW_FILENAME)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        log.info("Raw file already present at %s (skipping download).", dest)
        return dest

    log.info("Downloading dataset from %s", url)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    # Inject header row for downstream tools.
    body = resp.text.strip()
    header = ",".join(COLUMNS)
    dest.write_text(f"{header}\n{body}\n", encoding="utf-8")
    log.info("Wrote %d bytes to %s", dest.stat().st_size, dest)
    return dest


def main() -> int:
    try:
        path = download()
        log.info("OK | dataset at %s", path)
        return 0
    except Exception as exc:  # noqa: BLE001
        log.error("Download failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

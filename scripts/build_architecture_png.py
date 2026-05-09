"""Render the Mermaid diagram in docs/architecture.md to docs/architecture.png.

Uses the public mermaid.ink rendering service so no Node/mmdc install is needed.
"""
from __future__ import annotations

import base64
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "architecture.md"
DST = ROOT / "docs" / "architecture.png"


def main() -> int:
    md = SRC.read_text(encoding="utf-8")
    m = re.search(r"```mermaid\s*\n(.*?)\n```", md, re.DOTALL)
    if not m:
        print(f"No mermaid block found in {SRC}", file=sys.stderr)
        return 1
    src = m.group(1).strip()
    b = base64.urlsafe_b64encode(src.encode("utf-8")).decode("ascii")
    url = f"https://mermaid.ink/img/{b}?type=png&bgColor=FFFFFF"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    DST.write_bytes(data)
    print(f"Wrote {DST} ({len(data)/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

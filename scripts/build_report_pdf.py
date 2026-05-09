"""Convert docs/REPORT.docx to docs/MLOps_Assignment_Report_2025cs05056.pdf using Microsoft Word (COM).

Requires Microsoft Word installed locally and ``pywin32``. Produces a PDF
whose layout matches the source DOCX exactly because the conversion is
performed by Word itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

import win32com.client  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "REPORT.docx"
DST = ROOT / "docs" / "MLOps_Assignment_Report_2025cs05056.pdf"

WD_FORMAT_PDF = 17


def main() -> int:
    if not SRC.exists():
        sys.stderr.write(f"Source DOCX not found: {SRC}\n")
        sys.stderr.write("Run scripts/build_report_docx.py first.\n")
        return 1

    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    try:
        doc = word.Documents.Open(str(SRC), ReadOnly=True)
        try:
            doc.SaveAs2(str(DST), FileFormat=WD_FORMAT_PDF)
        finally:
            doc.Close(SaveChanges=False)
    finally:
        word.Quit()

    size_kb = DST.stat().st_size / 1024
    print(f"Wrote {DST} ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

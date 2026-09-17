import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.stages.s1_prep import prepare_document

DATA_DIR = Path(__file__).resolve().parent / "data"
TEXT_DIR = DATA_DIR / "text"
TEXT_DIR.mkdir(parents=True, exist_ok=True)


def load_pdf(pdf_path: str, doc_id: str) -> dict:
    """Extract text from a PDF, save it to data/text/<doc_id>.txt,
    and flag any pages that came back empty (possible scan/OCR need).

    Returns: {doc_id, pages, chars, warnings}
    """
    text_path = TEXT_DIR / f"{doc_id}.txt"
    result = prepare_document(pdf_path, str(text_path))

    warnings = []
    offsets = result.get("offsets", [])
    with open(text_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    for page_info in offsets:
        page_num = page_info["page"]
        start = page_info["start_char"]
        length = page_info["length"]
        page_text = full_text[start:start + length]
        # strip the "--- Page N ---" header s1_prep.py adds, then check content
        content_only = page_text.split("\n", 2)[-1].strip() if "\n" in page_text else page_text.strip()
        if len(content_only) < 10:  # near-empty page: likely scanned or blank
            warnings.append(f"Page {page_num} has little or no extractable text (possible scan)")

    return {
        "doc_id": doc_id,
        "pages": result["page_count"],
        "chars": result["char_count"],
        "warnings": warnings,
    }
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, UploadFile, File, HTTPException
from dotenv import load_dotenv

load_dotenv()

from loader import load_pdf
from runner import run_document
from llm_client import MODEL_CONFIG
from storage import DATA_DIR

app = FastAPI(title="RFP Benchmark Middleware")

UPLOAD_DIR = DATA_DIR / "uploads"
TEXT_DIR = DATA_DIR / "text"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TEXT_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def list_models():
    output = {}
    for key, cfg in MODEL_CONFIG.items():
        output[key] = {"name": cfg["name"], "base_url": cfg["base_url"]}
    return output


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    doc_id = Path(file.filename).stem.replace(" ", "_")
    pdf_path = UPLOAD_DIR / (doc_id + ".pdf")

    contents = await file.read()
    with open(pdf_path, "wb") as f:
        f.write(contents)

    result = load_pdf(str(pdf_path), doc_id)
    return result


@app.get("/documents/{doc_id}/text")
def get_document_text(doc_id: str):
    text_path = TEXT_DIR / (doc_id + ".txt")
    if not text_path.exists():
        msg = "No text found for doc_id: " + doc_id
        raise HTTPException(status_code=404, detail=msg)
    with open(text_path, "r", encoding="utf-8") as f:
        return {"doc_id": doc_id, "text": f.read()}


@app.post("/extract")
async def extract(doc_id: str, models: list[str] = None):
    text_path = TEXT_DIR / (doc_id + ".txt")
    if not text_path.exists():
        msg = "No prepared text for doc_id: " + doc_id + ". Upload it first."
        raise HTTPException(status_code=404, detail=msg)

    with open(text_path, "r", encoding="utf-8") as f:
        doc_text = f.read()

    result = await run_document(doc_id, doc_text, models)
    return result
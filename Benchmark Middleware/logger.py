"""One shared logger for the middleware.

Writes to the console (the uvicorn terminal) and to data/logs/middleware.log.

What is logged: model, doc_id, endpoint host, prompt size, schema mode, HTTP
status, latency, tokens, finish reason, parse result, and a short preview of
the model's answer.

What is NEVER logged: API keys, request headers, or the RFP text itself.
The full raw answer of every call is saved separately, to
data/raw/<doc_id>__<model>.txt (see storage.save_raw).
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "middleware.log"

# How much of a model answer to show in one log line.
PREVIEW_CHARS = 300

_FORMAT = "%(asctime)s %(levelname)-7s %(message)s"


def get_logger() -> logging.Logger:
    log = logging.getLogger("rfp")
    if log.handlers:  # already set up
        return log

    log.setLevel(logging.INFO)
    formatter = logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    log.addHandler(console)

    # 5 files of 2 MB each, oldest dropped
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=5,
                                       encoding="utf-8")
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)

    log.propagate = False  # do not double-print through uvicorn's root logger
    return log


def preview(text: str | None, limit: int = PREVIEW_CHARS) -> str:
    """One-line preview of a model answer: newlines flattened, cut to limit."""
    if not text:
        return "<empty>"
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + f"... (+{len(flat) - limit} chars)"

import json
import re
from typing import Dict, Any

class Normalizer:
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

    def normalize_record(self, raw_record: Dict[str, Any]) -> Dict[str, Any]:
        norm = {}
        for fid, cell in raw_record.items():
            if not isinstance(cell, dict):
                norm[fid] = cell
                continue
            status = cell.get("status", "not_stated")
            val = cell.get("value")
            ev = cell.get("evidence", [])
            if status == "not_stated" or val is None:
                norm[fid] = {"status": "not_stated", "value": None, "evidence": ev}
                continue
            if fid in ["F01", "F02"] and isinstance(val, dict) and "date" in val and val["date"]:
                m = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", str(val["date"]))
                if m:
                    val["date"] = f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
            norm[fid] = {"status": "found", "value": val, "evidence": ev}
        return norm

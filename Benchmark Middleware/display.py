"""Turn structured field values into the text the frontend shows.

The frontend only prints what it gets. It never has to know that a deadline
is three parts, or that a contact is a list of objects. Every field cell gets
a "display" string next to its exact "value".
"""

from __future__ import annotations

from typing import Any

NOT_FOUND = "Not found"


def _join(parts: list[str], sep: str = "; ") -> str:
    return sep.join(p for p in parts if p)


def _money(amount: Any, currency: Any) -> str:
    if amount is None:
        return ""
    try:
        text = f"{float(amount):,.0f}"
    except (TypeError, ValueError):
        text = str(amount)
    return f"{text} {currency}".strip()


def _deadline(v: dict) -> str:
    return _join([str(v.get("date") or ""), str(v.get("time") or ""),
                  str(v.get("tz_raw") or "")], " ")


def _contacts(v: list) -> str:
    out = []
    for c in v:
        if not isinstance(c, dict):
            out.append(str(c))
            continue
        name = _join([str(c.get("name") or ""), f"({c.get('title')})" if c.get("title") else ""], " ")
        out.append(_join([name, str(c.get("email") or ""), str(c.get("phone") or "")], ", "))
    return _join(out)


def _methods(v: list) -> str:
    out = []
    for m in v:
        if not isinstance(m, dict):
            out.append(str(m))
            continue
        label = str(m.get("portal_name") or m.get("url") or "")
        out.append(_join([str(m.get("method") or ""), label], ": "))
    return _join(out)


def _term(v: dict) -> str:
    base = ""
    if v.get("base_length") is not None:
        base = f"{v['base_length']} {v.get('base_unit') or ''}".strip()
    renewals = []
    for r in v.get("renewals") or []:
        if isinstance(r, dict):
            renewals.append(f"{r.get('count')} x {r.get('length')} {r.get('unit') or ''}".strip())
    text = _join([base, f"+ {_join(renewals, ', ')} renewal" if renewals else ""], " ")
    if v.get("total_max_years") is not None:
        text = f"{text} (max {v['total_max_years']} years)".strip()
    return text


def _scope(v: dict) -> str:
    summary = str(v.get("summary") or "")
    items = v.get("key_deliverables") or []
    if items:
        return summary + "\n" + "\n".join(f"- {i}" for i in items)
    return summary


def _criteria(v: list) -> str:
    out = []
    for c in v:
        if isinstance(c, dict):
            out.append(f"{c.get('criterion')}: {c.get('weight')} {c.get('unit') or ''}".strip())
        else:
            out.append(str(c))
    return _join(out)


def _threshold(v: dict) -> str:
    text = _join([str(v.get("value") or ""), str(v.get("unit") or "")], " ")
    if v.get("applies_to"):
        text = f"{text} ({v['applies_to']})".strip()
    return text


def _pricing(v: dict) -> str:
    model = str(v.get("pricing_model") or "")
    items = v.get("submission_requirements") or []
    if items:
        return model + "\n" + "\n".join(f"- {i}" for i in items)
    return model


def _insurance(v: list) -> str:
    out = []
    for c in v:
        if not isinstance(c, dict):
            out.append(str(c))
            continue
        basis = c.get("basis")
        basis_text = f"({basis.replace('_', ' ')})" if basis and basis != "unspecified" else ""
        out.append(_join([str(c.get("coverage_type") or ""),
                          _money(c.get("amount"), c.get("currency") or ""), basis_text], " "))
    return _join(out)


def _experience(v: list) -> str:
    out = []
    for e in v:
        if not isinstance(e, dict):
            out.append(str(e))
            continue
        head = str(e.get("requirement_type") or "")
        if e.get("years") is not None:
            head = f"{head} ({e['years']} yrs)".strip()
        out.append(_join([head, str(e.get("detail") or "")], ": "))
    return _join(out, "\n")


def _standards(v: list) -> str:
    out = []
    for s in v:
        if isinstance(s, dict):
            out.append(_join([str(s.get("standard") or ""), str(s.get("detail") or "")], ": "))
        else:
            out.append(str(s))
    return _join(out, "\n")


def _required_with_detail(v: dict, place_key: str) -> str:
    head = "Yes" if v.get("required") else "No"
    return _join([head, str(v.get(place_key) or ""), str(v.get("detail") or "")], " - ")


_RENDERERS = {
    "submission_deadline": _deadline,
    "questions_deadline": _deadline,
    "rfp_contact": _contacts,
    "submission_method": _methods,
    "contract_term": _term,
    "scope_of_deliverables": _scope,
    "evaluation_criteria": _criteria,
    "minimum_score_threshold": _threshold,
    "pricing_structure": _pricing,
    "insurance_requirements": _insurance,
    "vendor_experience": _experience,
    "data_security_requirements": _standards,
    "data_residency": lambda v: _required_with_detail(v, "location"),
    "demo_required": lambda v: _required_with_detail(v, "stage"),
}


def to_display(field_name: str, cell: dict | None) -> str:
    """One readable line (or short block) for a single field cell."""
    if not cell or cell.get("status") != "found":
        return NOT_FOUND

    value = cell.get("value")
    if value is None or value == [] or value == {}:
        return NOT_FOUND

    renderer = _RENDERERS.get(field_name)
    try:
        if renderer:
            text = renderer(value)
        elif isinstance(value, list):
            # plain lists of strings (mandatory requirements)
            text = "\n".join(f"- {i}" for i in value)
        else:
            text = str(value)
    except Exception:  # never let formatting break a result
        text = str(value)

    return text.strip() or NOT_FOUND


def add_display(fields: dict | None) -> dict | None:
    """Add a "display" string to every field cell, in place-safe fashion."""
    if not fields:
        return fields
    return {
        name: {**cell, "display": to_display(name, cell)} if isinstance(cell, dict) else cell
        for name, cell in fields.items()
    }

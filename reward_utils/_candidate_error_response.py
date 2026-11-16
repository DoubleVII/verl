"""Candidate response parsing/grounding ported from GRRM utils/candidate_error_response.py.

Keep envelope, normalized-key and Unicode grounding behavior in parity with GRRM.
No runtime dependency on that checkout.
"""

import json
import re
import unicodedata


def parse_response(text):
    """Read the last output fence or a final JSON object after free-form analysis.

    Do not salvage a malformed final fence using an earlier example/analysis object.
    JSON syntax remains JSON: no eval, invented fields, or partial list recovery.
    """
    if not isinstance(text, str):
        return None
    fences = list(re.finditer(r"```[ \t]*(?:json)?[ \t]*\r?\n(.*?)```", text, re.I | re.S))
    if fences:
        if "```" in text[fences[-1].end() :]:
            return None
        candidates = [fences[-1][1].strip()]
    else:
        if "```" in text:
            return None
        # Only accept an object ending at the response's end (allow Markdown whitespace).
        candidates = [text[i:].strip() for i, c in enumerate(text) if c == "{"]
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict) and any(key(k) == "spans" for k in value):
            return value
    return None


def key(value):
    return re.sub(r"[\s_\-]+", "", str(value)).casefold()


def fields(value):
    out = {}
    for k, v in value.items():
        k = key(k)
        if k in out:
            raise ValueError("colliding normalized keys")
        out[k] = v
    return out


def _fold(text):
    """Unicode/case/whitespace normalization with a map to original characters."""
    chars, offsets = [], []
    for i, char in enumerate(text):
        for c in unicodedata.normalize("NFKC", char).casefold():
            if not c.isspace():
                chars.append(c)
                offsets.append(i)
    return "".join(chars), offsets


def grounded_quote(quote, text):
    if not isinstance(quote, str) or not quote.strip():
        raise ValueError("missing quote")
    exact = [i for i in range(len(text)) if text.startswith(quote, i)]
    if len(exact) == 1:
        return quote, None
    if exact:
        raise ValueError("ambiguous quote; include more context")
    folded, positions = _fold(text)
    needle, _ = _fold(quote)
    if not needle:
        raise ValueError("empty normalized quote")
    starts = [i for i in range(len(folded)) if folded.startswith(needle, i)]
    if len(starts) != 1:
        raise ValueError("missing or ambiguous normalized quote")
    a = starts[0]
    start, end = positions[a], positions[a + len(needle) - 1] + 1
    # Never match half of a casefold expansion (e.g. s inside ß).
    if _fold(text[start:end])[0] != needle:
        raise ValueError("partial normalized character")
    canonical = text[start:end]
    if text.count(canonical) != 1:
        raise ValueError("ambiguous canonical quote")
    return canonical, dict(
        original_quote=quote, source_start=start, source_end=end, normalization="NFKC+casefold+ignore_whitespace"
    )


def validate_prediction(value, source):
    """Normalize compact score10 predictions; duplicate spans consume matching slots."""
    if not isinstance(value, dict):
        return None
    try:
        root = fields(value)
        if not isinstance(root.get("spans"), list):
            return None
        spans = []
        for raw in root["spans"]:
            if not isinstance(raw, dict):
                return None
            span = fields(raw)
            quote, _ = grounded_quote(span.get("sourcespan"), source)
            score = span.get("score")
            if isinstance(score, str):
                score = float(score.strip())
            if type(score) not in (int, float) or not 0 <= score <= 10 or int(score) != score:
                return None
            severity, reason = span.get("severity"), span.get("reason")
            if not isinstance(severity, str) or severity.strip().casefold() not in ("major", "minor"):
                return None
            if not isinstance(reason, str) or not reason.strip():
                return None
            spans.append(
                dict(source_span=quote, confidence=score / 10, severity=severity.strip().casefold(), reason=reason)
            )
        return {"spans": spans}
    except (ValueError, TypeError, KeyError, OverflowError):
        return None

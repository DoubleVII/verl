"""Search post-edit v2 wire protocol. Mirrored in GRRM utils and verl.tools.

Keep these two dependency-free copies identical; parity is covered by tests.
"""

import json
import re

VERSION = "search-post-edit-v2"
TOOL_NAME = "translation_search_batch"
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Search translation evidence for source spans in one search round.",
        "parameters": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "span_id": {"type": "integer", "minimum": 0},
                            "query": {"type": "string", "minLength": 1, "maxLength": 128},
                        },
                        "required": ["span_id", "query"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["queries"],
            "additionalProperties": False,
        },
    },
}


class ProtocolError(ValueError):
    pass


def loads(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ProtocolError("duplicate_json_key")
            result[key] = value
        return result

    try:
        return json.loads(
            text,
            object_pairs_hook=unique,
            parse_constant=lambda _: (_ for _ in ()).throw(ProtocolError("invalid_json")),
        )
    except (ValueError, TypeError) as exc:
        raise ProtocolError("invalid_json") from exc


def validate_budget(max_rounds):
    if type(max_rounds) is not int or not 0 <= max_rounds <= 2:
        raise ProtocolError("max_search_rounds_must_be_0_1_or_2")


def validate_queries(arguments, span_count):
    if not isinstance(arguments, dict) or set(arguments) != {"queries"}:
        raise ProtocolError("invalid_batch_parameters")
    queries = arguments["queries"]
    if not isinstance(queries, list) or not 1 <= len(queries) <= span_count:
        raise ProtocolError("query_count_exceeds_spans_or_empty")
    normalized, seen = [], set()
    for query in queries:
        if (
            not isinstance(query, dict)
            or set(query) - {"span_id", "query", "need_content"}
            or not {"span_id", "query"} <= set(query)
        ):
            raise ProtocolError("invalid_query_parameters")
        sid, text, content = query["span_id"], query["query"], query.get("need_content", False)
        if type(sid) is not int or not 0 <= sid < span_count or sid in seen:
            raise ProtocolError("invalid_or_duplicate_span_id")
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 128:
            raise ProtocolError("invalid_query_length")
        if re.search(r"(?:^|\s)-?(?:site|lang|ori_question|after|before):", text, re.I):
            raise ProtocolError("query_contains_search_operator")
        if type(content) is not bool:
            raise ProtocolError("need_content_must_be_boolean")
        seen.add(sid)
        normalized.append(dict(span_id=sid, query=text.strip(), need_content=content))
    return {"queries": normalized}


def require_analysis(text):
    """Validate the pilot's mandatory, nonempty Analysis section."""
    if not isinstance(text, str):
        raise ProtocolError("missing_analysis")
    match = re.fullmatch(r"# Analysis[ \t]*\r?\n(.+)", text.strip(), re.S)
    if not match or not match[1].strip():
        raise ProtocolError("missing_analysis")
    analysis = match[1].strip()
    if re.search(r"^# (?:Analysis|Final JSON)[ \t]*$", analysis, re.M) or re.search(
        r"</?(?:think|thinking)>", analysis
    ):
        raise ProtocolError("invalid_analysis_format")
    return analysis


def parse_turn(content, tool_calls=None):
    """Pilot envelope: mandatory Analysis, then native call or fenced Final JSON.

    Native APIs put Analysis in content and calls in tool_calls. Hermes contains
    both in one generated string. Never repair missing sections or invalid JSON.
    """
    if tool_calls is not None and not isinstance(tool_calls, list):
        raise ProtocolError("invalid_tool_calls")
    if tool_calls:
        if len(tool_calls) != 1:
            raise ProtocolError("expected_one_batch_call")
        require_analysis(content)
        call = tool_calls[0]
        if not isinstance(call, dict) or call.get("type", "function") != "function":
            raise ProtocolError("invalid_tool_call")
        function = call.get("function")
        if not isinstance(function, dict) or function.get("name") != TOOL_NAME:
            raise ProtocolError("unknown_tool")
        return "search", loads(function.get("arguments"))
    if not isinstance(content, str):
        raise ProtocolError("missing_assistant_content")
    text = content.strip()
    if "<tool_call>" in text or "</tool_call>" in text:
        matches = list(re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.S))
        if len(matches) != 1 or text.count("<tool_call>") != 1 or text.count("</tool_call>") != 1:
            raise ProtocolError("expected_one_complete_batch_call")
        match = matches[0]
        require_analysis(text[: match.start()])
        if text[match.end() :].strip():
            raise ProtocolError("unexpected_text_after_tool_call")
        call = loads(match[1])
        if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
            raise ProtocolError("invalid_tool_call")
        if call["name"] != TOOL_NAME:
            raise ProtocolError("unknown_tool")
        return "search", call["arguments"]
    if not re.match(r"^# Analysis[ \t]*\r?\n", text):
        raise ProtocolError("missing_analysis")
    match = re.fullmatch(
        r"(# Analysis[ \t]*\r?\n.+?)\r?\n# Final JSON[ \t]*\r?\n"
        r"```json[ \t]*\r?\n(.*?)\r?\n```",
        text,
        re.S,
    )
    if not match:
        raise ProtocolError("invalid_final_envelope")
    require_analysis(match[1])
    final = loads(match[2])
    if (
        not isinstance(final, dict)
        or set(final) != {"translation"}
        or not isinstance(final["translation"], str)
        or not final["translation"].strip()
    ):
        raise ProtocolError("invalid_final_translation")
    return "finish", final["translation"]


def compact_evidence(payload, span_ids, max_chars=24000):
    """Validate service responses and shrink structured evidence without cutting JSON."""
    if not isinstance(payload, dict) or not isinstance(payload.get("results_by_span"), list):
        raise ValueError("invalid_search_response")
    rows = payload["results_by_span"]
    if len(rows) != len(span_ids):
        raise ValueError("search_response_span_mismatch")
    evidence, metadata = [], []
    limits = {"title": 300, "url": 1000, "snippet": 500, "summary": 1600, "content": 3000}
    truncated = False
    for row, sid in zip(rows, span_ids, strict=False):
        if not isinstance(row, dict) or type(row.get("span_id")) is not int or row["span_id"] != sid:
            raise ValueError("search_response_span_mismatch")
        if row.get("error") is not None:
            raise ValueError("search_service_query_failed")
        if not isinstance(row.get("results"), list):
            raise ValueError("invalid_search_results")
        results = []
        for result in row["results"]:
            if not isinstance(result, dict):
                raise ValueError("invalid_search_result")
            item = {}
            for key, limit in limits.items():
                value = result.get(key)
                if value is not None:
                    if not isinstance(value, str):
                        raise ValueError("invalid_search_result_text")
                    item[key] = value[:limit]
                    truncated |= len(value) > limit
            for key in ("authority", "rerank_score"):
                if type(result.get(key)) in (int, float):
                    item[key] = result[key]
            results.append(item)
        evidence.append({"span_id": sid, "results": results, "filtered_count": row.get("filtered_count", 0)})
        metadata.append({"span_id": sid, "meta": row.get("meta", {})})
    output = {"results_by_span": evidence, "truncated": truncated}
    while len(json.dumps(output, ensure_ascii=False, allow_nan=False)) > max_chars:
        longest = max(evidence, key=lambda row: len(json.dumps(row["results"], ensure_ascii=False)))
        if not longest["results"]:
            raise ValueError("search_response_budget_too_small")
        longest["results"].pop()
        output["truncated"] = True
    return output, metadata

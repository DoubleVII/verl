# Copyright 2026 Individual Contributor: Yangs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Single-sample generative reward function for machine translation."""

from __future__ import annotations

import json
import math
from typing import Any

import aiohttp


LANGUAGE_NAMES = {
    "ar": "Arabic",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "th": "Thai",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "zh": "Chinese",
}

PROMPT_TEMPLATE = """Given a source text in {source_language} and a translation text in {target_language}. Perform a step by step analysis of translation quality and assign a score on a scale from 0 to 10.
Source text:
```
{source_text}
```

Translation text:
```
{translation_text}
```
{notes}"""

NOTES_TEMPLATE = """

You may refer to the following notes, if helpful, when evaluating the translation.

Notes:
```
{notes}
```
"""


def _extract_translation(response: str, extractor_type: str) -> str | None:
    response = response.strip()
    if not response:
        return None
    if extractor_type == "none":
        return response
    if extractor_type == "line":
        return response.splitlines()[-1].strip() or None
    if extractor_type == "oneline":
        return response if "\n" not in response else None
    if extractor_type == "codeblock":
        if response.count("```") != 2 or not response.endswith("```"):
            return None
        block = response[:-3]
        block = block[block.rfind("```") + 3 :]
        if "\n" in block:
            block = block.split("\n", 1)[1]
        return block.strip() or None
    raise ValueError(f"Unknown extractor_type: {extractor_type!r}")


def _get_language_pair(extra_info: dict[str, Any]) -> tuple[str, str]:
    if "src_lang" in extra_info and "trg_lang" in extra_info:
        return str(extra_info["src_lang"]), str(extra_info["trg_lang"])
    if "lang_pair" in extra_info:
        parts = str(extra_info["lang_pair"]).split("-", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
    raise ValueError("extra_info must contain src_lang/trg_lang or lang_pair")


def _language_matches(text: str, target_language: str) -> bool:
    try:
        from lingua import Language, LanguageDetectorBuilder
    except ImportError as exc:
        raise ImportError(
            "enable_language_detection=True requires `lingua-language-detector`"
        ) from exc

    language = getattr(Language, LANGUAGE_NAMES.get(target_language, "").upper(), None)
    if language is None:
        return True
    supported_languages = [
        getattr(Language, name.upper())
        for name in LANGUAGE_NAMES.values()
        if hasattr(Language, name.upper())
    ]
    detector = LanguageDetectorBuilder.from_languages(*supported_languages).build()
    return detector.detect_language_of(text) == language


def _extract_score(output: str) -> float | None:
    if not output.strip():
        return None
    try:
        score = float(output.strip().splitlines()[-1].strip())
    except ValueError:
        return None
    return score if 0 <= score <= 10 else None


async def _chat_complete(router_address: str, request: dict[str, Any]) -> str:
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            f"http://{router_address}/v1/chat/completions", json=request
        ) as response:
            response.raise_for_status()
            payload = json.loads(await response.text())
    return payload["choices"][0]["message"]["content"]


def _result(
    score: float,
    status: str,
    *,
    raw_score: float = math.nan,
    response: str = "",
    overlong_penalty: float = 0.0,
) -> dict[str, Any]:
    # RewardLoopManager requires every sample in a batch to expose the same
    # reward-extra-info keys when it assembles numpy arrays.
    return {
        "score": score,
        "genrm_status": status,
        "genrm_raw_score": raw_score,
        "genrm_response": response,
        "genrm_overlong_penalty": overlong_penalty,
    }


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any],
    reward_router_address: str,
    reward_model_tokenizer: Any,
    *,
    model_name: str,
    extractor_type: str = "codeblock",
    score_scale_factor: float = 0.01,
    default_reward: float = -0.04,
    enable_language_detection: bool = False,
    max_prompt_length: int = 2048,
    max_tokens: int = 8192,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = -1,
    overlong_buffer_enable: bool = False,
    max_response_length: int = 3072,
    overlong_buffer_length: int = 2048,
    overlong_penalty_factor: float = 0.04,
) -> dict[str, Any]:
    """Score one rollout response with the configured generative RM."""
    translation = _extract_translation(solution_str, extractor_type)
    if translation is None:
        return _result(default_reward, "invalid_translation_format")

    source_language, target_language = _get_language_pair(extra_info)
    if enable_language_detection and not _language_matches(translation, target_language):
        return _result(default_reward, "language_mismatch")

    notes = str(extra_info.get("notes") or "").strip()
    prompt = PROMPT_TEMPLATE.format(
        source_language=LANGUAGE_NAMES.get(source_language, source_language),
        target_language=LANGUAGE_NAMES.get(target_language, target_language),
        source_text=extra_info["src_text"],
        translation_text=translation,
        notes=NOTES_TEMPLATE.format(notes=notes) if notes else "",
    )
    messages = [{"role": "user", "content": prompt}]
    prompt_ids = reward_model_tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    if len(prompt_ids) > max_prompt_length:
        return _result(default_reward, "prompt_too_long")

    request = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if top_k >= 0:
        request["top_k"] = top_k
    genrm_response = await _chat_complete(reward_router_address, request)
    raw_score = _extract_score(genrm_response)
    if raw_score is None:
        return _result(default_reward, "invalid_reward_format", response=genrm_response)

    penalty = 0.0
    if overlong_buffer_enable and overlong_buffer_length > 0:
        response_length = len(reward_model_tokenizer.encode(solution_str, add_special_tokens=False))
        threshold = max_response_length - overlong_buffer_length
        excess = max(response_length - threshold, 0)
        penalty = min(excess / overlong_buffer_length, 1.0) * overlong_penalty_factor

    return _result(
        raw_score * score_scale_factor - penalty,
        "ok",
        raw_score=raw_score,
        response=genrm_response,
        overlong_penalty=penalty,
    )

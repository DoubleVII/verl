from typing import Any, List

import numpy as np

try:
    from .config import LANG_MAP, candidate_identifiers
except ImportError:
    from reward_utils.config import LANG_MAP, candidate_identifiers

try:
    from verl.interactions.gqm_post_edit_interaction import GQM_POST_EDIT_PROMPT
except ImportError:
    GQM_POST_EDIT_PROMPT = (
        "Using the source text, candidate translations, and evaluation above, provide the final improved translation "
        "in the target language. Include a concise step-by-step analysis, then output only the final translation in a "
        "single Markdown code block."
    )


def _render_reward_model_prompt(prompt: Any, tokenizer: Any) -> str:
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    if isinstance(prompt, dict) and "prompt_token_ids" in prompt:
        return tokenizer.decode(prompt["prompt_token_ids"], skip_special_tokens=False)
    if isinstance(prompt, (list, tuple)):
        return tokenizer.decode(list(prompt), skip_special_tokens=False)
    return str(prompt)


def gqm_post_edit_teacher_prompt_constructor(
    prompt: Any,
    response: Any,
    tokenizer: Any,
    sample: dict[str, Any] | None = None,
    config: Any | None = None,
    post_edit_prompt: str = GQM_POST_EDIT_PROMPT,
) -> list[dict[str, str]]:
    """Build a post-edit teacher prompt from generic reward-model metadata."""
    prompt_text = _render_reward_model_prompt(prompt, tokenizer).strip()
    if isinstance(response, np.ndarray):
        response = response.tolist()
    response_text = str(response).strip()
    if not prompt_text:
        raise ValueError("Reward model prompt is missing or empty.")
    if not response_text:
        raise ValueError("Reward model response is missing or empty.")
    return [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": response_text},
        {"role": "user", "content": post_edit_prompt},
    ]


SINGLE_PROMPT_TEMPLATE = """Given a source text in {} and a translation text in {}. Perform a step by step analysis of translation quality and assign a score on a scale from 0 to 10.
Source text:
```
{}
```

Translation text:
```
{}
```
"""

SINGLE_NOTES_PROMPT_TEMPLATE = """

You may refer to the following notes, if helpful, when evaluating the translation.

Notes:
```
{}
```
"""


def single_get_prompt(src_text, mt_text, src_lang, tgt_lang, notes=None):
    if len(src_lang) == 2:
        src_lang = LANG_MAP[src_lang]
    if len(tgt_lang) == 2:
        tgt_lang = LANG_MAP[tgt_lang]
    base = SINGLE_PROMPT_TEMPLATE.format(src_lang, tgt_lang, src_text, mt_text)
    if notes is not None:
        notes = notes.strip()
        if notes:
            base += SINGLE_NOTES_PROMPT_TEMPLATE.format(notes)
    return base


GROUP_TASK_FORMAT = {
    "score": "Finally, score the candidates with integer scores on a scale from 0 to 10.",
    "ranking": "Finally, rank the candidates in order of quality from best to worst.",
    "ranking_score": "Finally, rank and score the candidates with integer scores on a scale from 0 to 10.",
}

GROUP_OUTPUT_EXAMPLE = {
    "score": "Output the scores on the last line, for example: `A: 4, B: 9, C: 7, D: 9`.",
    "ranking": "Output the rankings in descending order on the last line, for example: `B > A = D > C`.",
    "ranking_score": "At the end section, first output the rankings in descending order, for example: `B > A = D > C`. Then, on the last line, output the scores, for example: `B: 9, A: 7, D: 7, C: 2`.",
}

GROUP_PROMPT_TEMPLATE = """Given a source text in {} and multiple translation candidates in {}. Perform a step by step analysis and comparison of the translation quality for the candidates. {}

Source text:
```
{}
```

{}{}"""

CANDIDATE_PROMPT = """Translation {}:
```
{}
```
"""


def _group_get_task_prompt(prompt_format: str, add_example: bool = False) -> str:
    if prompt_format not in GROUP_TASK_FORMAT:
        raise ValueError(f"prompt_format must be one of {GROUP_TASK_FORMAT.keys()}")
    task_prompt = GROUP_TASK_FORMAT[prompt_format]
    if add_example:
        task_prompt += f" {GROUP_OUTPUT_EXAMPLE[prompt_format]}"
    return task_prompt


NOTES_PROMPT_TEMPLATE = """

You may refer to the following notes, if helpful, when evaluating the translations.

Notes:
```
{}
```
"""


REFERENCE_PROMPT_TEMPLATE = """

You may refer to the following reference, if helpful, when evaluating the translations.

{} reference:
```
{}
```
"""


def _build_notes_prompt(notes=None):
    if notes is None:
        return ""
    notes = notes.strip()
    if not notes:
        return ""
    return NOTES_PROMPT_TEMPLATE.format(notes)


def _build_reference_prompt(ref_text=None, ref_lang=None):
    if ref_text is None or ref_lang is None:
        return ""
    ref_text = ref_text.strip()
    ref_lang = ref_lang.strip()
    if not ref_text or not ref_lang:
        return ""
    if len(ref_lang) == 2 and ref_lang in LANG_MAP:
        ref_lang = LANG_MAP[ref_lang]
    return REFERENCE_PROMPT_TEMPLATE.format(ref_lang, ref_text)


def get_GQM_prompt(
    source_lang: str,
    target_lang: str,
    source_text: str,
    mt_texts: List[str],
    prompt_format: str,
    add_example: bool = False,
    notes: str = None,
    ref_text: str = None,
    ref_lang: str = None,
) -> str:
    if len(mt_texts) == 1:
        raise ValueError("Only support multiple candidates.")
    if len(mt_texts) > len(candidate_identifiers):
        raise ValueError(f"Only support {len(candidate_identifiers)} candidates.")
    if len(source_lang) == 2 and source_lang in LANG_MAP:
        source_lang = LANG_MAP[source_lang]
    if len(target_lang) == 2 and target_lang in LANG_MAP:
        target_lang = LANG_MAP[target_lang]
    task_prompt = _group_get_task_prompt(prompt_format, add_example)
    candidate_prompts = "".join([CANDIDATE_PROMPT.format(candidate_identifiers[i], mt_texts[i]) for i in range(len(mt_texts))])
    reference_prompt = _build_reference_prompt(ref_text, ref_lang)
    notes_prompt = _build_notes_prompt(notes)
    return GROUP_PROMPT_TEMPLATE.format(source_lang, target_lang, task_prompt, source_text, candidate_prompts, reference_prompt + notes_prompt)


def _seedx_build_prompt(src_text: str, mt_text: str, src_lang: str, trg_lang: str) -> str:
    src_display = LANG_MAP[src_lang] if len(src_lang) == 2 and src_lang in LANG_MAP else src_lang
    trg_display = LANG_MAP[trg_lang] if len(trg_lang) == 2 and trg_lang in LANG_MAP else trg_lang
    trg_tag = f" <{trg_lang}>" if len(trg_lang) == 2 else ""
    prompt = (
        f"Translate the following {src_display} sentence into {trg_display}:\n"
        f"{src_text}{trg_tag}"
    )
    return prompt


def _vanilla_rm_build_prompt(tokenizer, src_lang: str, trg_lang: str, src_text: str, mt_text: str, chat_template: bool = True) -> str:
    """Build the full prompt + response text with EOS for value-head reward model scoring."""
    src_display = LANG_MAP[src_lang] if len(src_lang) == 2 and src_lang in LANG_MAP else src_lang
    trg_display = LANG_MAP[trg_lang] if len(trg_lang) == 2 and trg_lang in LANG_MAP else trg_lang
    prompt = f"Translate the following text from {src_display} into {trg_display}:\n{src_display}: {src_text}\n{trg_display}:"
    if chat_template:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt += " "
    full_prompt = f"{prompt}{mt_text}{tokenizer.eos_token}"
    return full_prompt

from typing import List

try:
    from .config import LANG_MAP, candidate_identifiers
except ImportError:
    from reward_utils.config import LANG_MAP, candidate_identifiers


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


def single_get_prompt(src_text, mt_text, src_lang, tgt_lang):
    if len(src_lang) == 2:
        src_lang = LANG_MAP[src_lang]
    if len(tgt_lang) == 2:
        tgt_lang = LANG_MAP[tgt_lang]
    return SINGLE_PROMPT_TEMPLATE.format(src_lang, tgt_lang, src_text, mt_text)


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

{}"""

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


def group_get_prompt(source_lang: str, target_lang: str, source_text: str, mt_texts: List[str], prompt_format: str, add_example: bool = False) -> str:
    if len(mt_texts) == 1:
        raise ValueError("Only support multiple candidates.")
    if len(mt_texts) > len(candidate_identifiers):
        raise ValueError(f"Only support {len(candidate_identifiers)} candidates.")
    task_prompt = _group_get_task_prompt(prompt_format, add_example)
    candidate_prompts = "".join([CANDIDATE_PROMPT.format(candidate_identifiers[i], mt_texts[i]) for i in range(len(mt_texts))])
    return GROUP_PROMPT_TEMPLATE.format(source_lang, target_lang, task_prompt, source_text, candidate_prompts)


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

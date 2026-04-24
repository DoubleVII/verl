import asyncio

import openai

from reward_utils.config import LANG_MAP, candidate_identifiers
from reward_utils.helpers import _line_extractor, _block_extractor, _one_line_extractor, group_extract_scores

Output_example = {
    "score": "Output the scores on the last line, for example: `A: 4, B: 9, C: 7, D: 9`.",
    "ranking": "Output the rankings in descending order on the last line, for example: `B > A = D > C`.",
    "ranking_score": "At the end section, first output the rankings in descending order, for example: `B > A = D > C`. Then, on the last line, output the scores, for example: `B: 9, A: 7, D: 7, C: 2`.",
}


Task_format = {
    "score": "Finally, score the candidates with integer scores on a scale from 0 to 10.",
    "ranking": "Finally, rank the candidates in order of quality from best to worst.",
    "ranking_score": "Finally, rank and score the candidates with integer scores on a scale from 0 to 10.",
}


GQM_prompt_template = """Given a source text in {source_lang} and multiple translation candidates in {target_lang}. Perform a step by step analysis and comparison of the translation quality for the candidates. {task_prompt}

Source text:
```
{source_text}
```

{candidate_prompts}{notes_prompt}"""


notes_prompt_template = """

You may refer to the following notes, if helpful, when evaluating the translations.

Notes:
```
{notes}
```
"""


candidate_prompt = """Translation {}:
```
{}
```
"""


def get_task_prompt(prompt_format: str, add_example: bool = False):
    if prompt_format not in Task_format:
        raise ValueError(f"prompt_format must be one of {Task_format.keys()}")
    task_prompt = Task_format[prompt_format]
    if add_example:
        task_prompt += f" {Output_example[prompt_format]}"
    return task_prompt


def build_notes_prompt(notes: str = None) -> str:
    if notes is None:
        return ""
    notes = notes.strip()
    if not notes:
        return ""
    return notes_prompt_template.format(notes=notes)


def get_GQM_with_notes_prompt(
    source_lang,
    target_lang,
    source_text,
    mt_texts,
    prompt_format: str,
    add_example: bool = False,
    notes: str = None,
):
    if len(source_lang) == 2:
        source_lang = LANG_MAP[source_lang]
    if len(target_lang) == 2:
        target_lang = LANG_MAP[target_lang]
    if len(mt_texts) == 1:
        raise ValueError("Only support multiple candidates.")
    if len(mt_texts) > len(candidate_identifiers):
        raise ValueError(f"Only support {len(candidate_identifiers)} candidates.")

    task_prompt = get_task_prompt(prompt_format, add_example)

    candidate_prompts = "".join(
        candidate_prompt.format(candidate_identifiers[i], mt_texts[i])
        for i in range(len(mt_texts))
    )

    notes_prompt = build_notes_prompt(notes)

    return GQM_prompt_template.format(
        source_lang=source_lang,
        target_lang=target_lang,
        task_prompt=task_prompt,
        source_text=source_text,
        candidate_prompts=candidate_prompts,
        notes_prompt=notes_prompt,
    )



def extract_mt(response: str, extractor_type: str = "codeblock"):
    if extractor_type == "line":
        return _line_extractor(response)
    elif extractor_type == "codeblock":
        return _block_extractor(response)
    elif extractor_type == "oneline":
        return _one_line_extractor(response)
    elif extractor_type == "none":
        return response.strip()
    else:
        raise ValueError(f"extractor_type: {extractor_type}")



async def group_post_edit_score_reward_fn(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    api_base_url: str,
    api_key: str = "EMPTY",
    rm_max_candidates: int = 4,
    rm_sampling_temp: float = 1,
    rm_sampling_top_p: float = 1,
    extractor_type: str = "codeblock",
    default_reward: float = -1.0,
    reward_scale: float = 1.0,
    prompt_format: str = "ranking_score",
    add_example: bool = False,
    rm_model_name: str = None,
    rm_max_tokens: int = 8192,
    rm_retry: int = 2,
    enable_language_detection: bool = False,
) -> dict:
    reward_out = {"score": default_reward, "valid_answer": 0, "pt_mt_score": 0, "baseline_score": 0}
    pe_mt_text = extract_mt(solution_str, extractor_type)
    if pe_mt_text is None:
        return reward_out

    # extract fields from extra_info
    src_text = extra_info.get("src_text", ground_truth)
    mt_texts = extra_info.get("mt_texts", [])
    notes = extra_info.get("notes", None)
    src_lang = extra_info.get("src_lang", "English")
    trg_lang = extra_info.get("trg_lang", "Chinese")

    # language detection check
    if enable_language_detection:
        try:
            from reward_utils.language_detector import is_language_match
        except ImportError:
            from .language_detector import is_language_match
        if not is_language_match(pe_mt_text, trg_lang):
            return reward_out

    # truncate baselines to respect rm_max_candidates
    if 1 + len(mt_texts) > rm_max_candidates:
        mt_texts = mt_texts[: rm_max_candidates - 1]

    # pe_mt is placed last to avoid order bias in the reward model
    all_mt_texts = mt_texts + [pe_mt_text]
    pe_mt_index = len(all_mt_texts) - 1
    num_candidates = len(all_mt_texts)

    if num_candidates < 2:
        return reward_out

    # build GQM prompt (notes included unless empty/None)
    prompt = get_GQM_with_notes_prompt(
        source_lang=src_lang,
        target_lang=trg_lang,
        source_text=src_text,
        mt_texts=all_mt_texts,
        prompt_format=prompt_format,
        add_example=add_example,
        notes=notes,
    )

    # call vLLM with retry
    client = openai.AsyncOpenAI(base_url=api_base_url, api_key=api_key)
    messages = [{"role": "user", "content": prompt}]

    scores = None
    for attempt in range(rm_retry):
        try:
            response = await client.chat.completions.create(
                model=rm_model_name,
                messages=messages,
                temperature=rm_sampling_temp,
                top_p=rm_sampling_top_p,
                max_tokens=rm_max_tokens,
            )
            output_text = response.choices[0].message.content
        except Exception as e:
            print(f"[post_edit_reward_remote] API call failed (attempt {attempt + 1}/{rm_retry}): {e}")
            continue

        scores = group_extract_scores(output_text, prompt_format, num_candidates)
        if scores is not None:
            break
        print(f"[post_edit_reward_remote] Score extraction failed (attempt {attempt + 1}/{rm_retry})")

    if scores is None:
        return reward_out

    # compute relative reward
    pe_mt_score = scores[pe_mt_index]
    baseline_scores = scores[:pe_mt_index]
    mean_all = sum(scores) / len(scores)
    relative_reward = pe_mt_score - mean_all
    final_reward = reward_scale * relative_reward

    baseline_mean = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0
    return {
        "score": final_reward,
        "valid_answer": 1,
        "pt_mt_score": pe_mt_score,
        "baseline_score": baseline_mean,
    }


def batch_group_post_edit_score_reward_fn(
    data_sources: list,
    solution_strs: list,
    ground_truths: list,
    extra_infos: list,
    api_base_url: str,
    api_key: str = "EMPTY",
    rm_max_candidates: int = 4,
    rm_sampling_temp: float = 1,
    rm_sampling_top_p: float = 1,
    extractor_type: str = "codeblock",
    default_reward: float = -1.0,
    reward_scale: float = 1.0,
    prompt_format: str = "ranking_score",
    add_example: bool = False,
    rm_model_name: str = None,
    rm_max_tokens: int = 8192,
    rm_retry: int = 2,
    enable_language_detection: bool = False,
) -> list[dict]:
    if rm_model_name is None:
        raise ValueError("rm_model_name must be provided.")
    async def _run_all():
        tasks = [
            group_post_edit_score_reward_fn(
                data_source=data_sources[i],
                solution_str=solution_strs[i],
                ground_truth=ground_truths[i],
                extra_info=extra_infos[i],
                api_base_url=api_base_url,
                api_key=api_key,
                rm_max_candidates=rm_max_candidates,
                rm_sampling_temp=rm_sampling_temp,
                rm_sampling_top_p=rm_sampling_top_p,
                extractor_type=extractor_type,
                default_reward=default_reward,
                reward_scale=reward_scale,
                prompt_format=prompt_format,
                add_example=add_example,
                rm_model_name=rm_model_name,
                rm_max_tokens=rm_max_tokens,
                rm_retry=rm_retry,
                enable_language_detection=enable_language_detection,
            )
            for i in range(len(solution_strs))
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    loop = asyncio.new_event_loop()
    try:
        results = loop.run_until_complete(_run_all())
    finally:
        loop.close()

    default_out = {"score": default_reward, "valid_answer": 0, "pt_mt_score": 0, "baseline_score": 0}
    reward_out_list = []
    for result in results:
        if isinstance(result, Exception):
            print(f"[post_edit_reward_remote] Item failed: {result}")
            reward_out_list.append(dict(default_out))
        elif isinstance(result, dict):
            reward_out_list.append(result)
        else:
            reward_out_list.append(dict(default_out))
    return reward_out_list

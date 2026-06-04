import asyncio

import openai

try:
    from .prompts import get_GQM_prompt
except ImportError:
    from reward_utils.prompts import get_GQM_prompt

from reward_utils.helpers import _line_extractor, _block_extractor, _one_line_extractor, group_extract_scores



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
    prompt_format: str = "ranking_score",
    add_example: bool = False,
    rm_model_name: str = None,
    rm_max_tokens: int = 8192,
    rm_retry: int = 2,
    enable_language_detection: bool = False,
    rm_timeout: float = 60.0,
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
    prompt = get_GQM_prompt(
        source_lang=src_lang,
        target_lang=trg_lang,
        source_text=src_text,
        mt_texts=all_mt_texts,
        prompt_format=prompt_format,
        add_example=add_example,
        notes=notes,
    )

    # call vLLM with retry
    client = openai.AsyncOpenAI(base_url=api_base_url, api_key=api_key, timeout=rm_timeout)
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

    baseline_mean = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0
    return {
        "score": relative_reward,
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
    prompt_format: str = "ranking_score",
    add_example: bool = False,
    rm_model_name: str = None,
    rm_max_tokens: int = 8192,
    rm_retry: int = 2,
    enable_language_detection: bool = False,
    rm_timeout: float = 60.0,
    max_concurrent: int = 8,
) -> list[dict]:
    if rm_model_name is None:
        raise ValueError("rm_model_name must be provided.")
    async def _run_all():
        sem = asyncio.Semaphore(max_concurrent)

        async def _bounded(i):
            async with sem:
                return await group_post_edit_score_reward_fn(
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
                    prompt_format=prompt_format,
                    add_example=add_example,
                    rm_model_name=rm_model_name,
                    rm_max_tokens=rm_max_tokens,
                    rm_retry=rm_retry,
                    enable_language_detection=enable_language_detection,
                    rm_timeout=rm_timeout,
                )

        tasks = [_bounded(i) for i in range(len(solution_strs))]
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

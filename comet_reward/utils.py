from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Iterable
import requests
import time
import re


def score_normalize(score: float, min_score: float, max_score: float) -> float:
    """
    clip score
    """
    return min(max(score, min_score), max_score)


import time
from typing import List, Literal
import requests


THINKING_TAGS = ["<analysis>", "</analysis>"]
OLD_THINKING_TAGS = ["<think>", "</think>"]
TRANSLATION_TAGS = ["<translation>", "</translation>"]
OLD_TRANSLATION_TAGS = ["<translate>", "</translate>"]

TRANSLATION_REGEX_STR = r"<translation>(.*?)</translation>"
OLD_TRANSLATION_REGEX_STR = r"<translate>(.*?)</translate>"

import os

# get env var
use_old_tag = os.getenv("TRANSLATION_PROMPT_VERSION") == "v1"
if use_old_tag:
    THINKING_TAGS = OLD_THINKING_TAGS
    TRANSLATION_REGEX_STR = OLD_TRANSLATION_REGEX_STR
    TRANSLATION_TAGS = OLD_TRANSLATION_TAGS


def get_rewards_from_server(
    server_url: str,
    response_texts: List[str],
    src_langs: List[str],
    trg_langs: List[str],
    src_texts: List[str] = None,
    ref_texts: List[str] = None,
    batch_size: int = 128,
) -> List[float]:
    r"""
    Gets reward scores from the API server by splitting requests into batches.
    """

    total_items = len(response_texts)
    all_rewards = []

    # 处理单个批次的请求（包含重试逻辑）
    def _request_single_batch(batch_indices):
        batch_payload = {
            "ref_list": [ref_texts[i] for i in batch_indices] if ref_texts else None,
            "src_list": [src_texts[i] for i in batch_indices] if src_texts else None,
            "mt_list": [response_texts[i] for i in batch_indices],
            "src_langs": [src_langs[i] for i in batch_indices],
            "trg_langs": [trg_langs[i] for i in batch_indices],
        }

        print(f"[Request] remote reward request batch size: {len(batch_indices)}")
        max_retries = 5
        backoff_factor = 5

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    server_url,
                    json=batch_payload,
                    headers={"Content-Type": "application/json"},
                    timeout=40,
                )
                response.raise_for_status()
                return response.json()["scores"]
            except Exception as e:
                if attempt == max_retries - 1:
                    # 仅在最终失败时保存错误批次
                    with open("/opt/tiger/error.json", "w") as f:
                        import json

                        json.dump(batch_payload, f, ensure_ascii=False, indent=4)
                    raise RuntimeError(
                        f"Batch request failed after {max_retries} attempts"
                    ) from e
                wait_time = backoff_factor * (2**attempt)
                print(
                    f"Request failed (batch size {len(batch_indices)}). Retrying in {wait_time} seconds..."
                )
                time.sleep(wait_time)

    # 分批处理所有数据
    for start_idx in range(0, total_items, batch_size):
        end_idx = min(start_idx + batch_size, total_items)
        batch_indices = list(range(start_idx, end_idx))
        batch_rewards = _request_single_batch(batch_indices)
        all_rewards.extend(batch_rewards)

    return all_rewards


def format_checker(solution_str: str, forbidden_tags: list[str]) -> bool:
    for tag in forbidden_tags:
        if tag in solution_str:
            return False
    if solution_str == "":
        return False
    return True

def presence_checker(solution_str: str, presence_tags: list[str], force_once: bool = False) -> bool:
    for tag in presence_tags:
        presence_count = solution_str.count(tag)
        if presence_count == 0:
            return False
        if presence_count > 1 and force_once:
            return False
    return True

def extract_thinking_translation(solution_str: str):
    """Extracts the final answer from the model's response string.

    Args:
        solution_str: Raw response string from the language model

    Returns:
        extracted_answer
    """
    presence_tags = THINKING_TAGS + TRANSLATION_TAGS
    if not presence_checker(solution_str, presence_tags, force_once=True):
        return None
    # Extract final answer using XML-style tags
    answer_pattern = TRANSLATION_REGEX_STR
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))

    if not matches:
        # print("[Error] No valid answer tags found")
        return None

    final_answer = matches[-1].group(1).strip()

    if format_checker(
        final_answer, presence_tags
    ):
        return final_answer
    return None

def extract_last_line(solution_str: str):
    """Extracts the last line from the model's response string.

    Args:
        solution_str: Raw response string from the language model

    Returns:
        extracted_last_line
    """
    lines = solution_str.strip().split("\n")
    return lines[-1]


def extract_markdown(solution_str: str):
    """Extracts the last line from markdown response.
    """
    presence_tags = ["# Analysis", "# Translation"]
    if not presence_checker(solution_str, presence_tags, force_once=True):
        return None
    
    lines = solution_str.strip().split("\n")
    if lines[0] != presence_tags[0]:
        return None
    if lines[-2] != presence_tags[1]:
        return None
    return lines[-1]

def extract_no_thinking_translation(solution_str: str):
    """Extracts the final answer from the model's response string.

    Args:
        solution_str: Raw response string from the language model

    Returns:
        extracted_answer
    """
    solution_str = solution_str.strip()
    presence_tags = TRANSLATION_TAGS
    if not presence_checker(solution_str, presence_tags, force_once=True):
        return None
    if not solution_str.startswith(presence_tags[0]):
        return None
    if not solution_str.endswith(presence_tags[1]):
        return None

    # Extract final answer using XML-style tags
    answer_pattern = TRANSLATION_REGEX_STR
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))

    if not matches:
        # print("[Error] No valid answer tags found")
        return None

    final_answer = matches[-1].group(1).strip()

    if format_checker(
        final_answer, presence_tags
    ):
        return final_answer
    return None

def compute_score(
    data_sources,
    solution_strs,
    ground_truths,
    extra_infos=None,
    batch_size=128,
    server_url: str = None,
    use_extract_translation: str = "none",
    use_bleu_penalty: bool = False,
    use_length_penalty: bool = False,
    filter_max_len: int = 1024,
    penalty_buffer_len: int = 256,
    short_penalty_buffer_len: int = 0,
    response_lengths: list[int] = None,
    en_proxy_reward: bool = False,
    normalize_type: Literal["zero2one", "scale", "none"] = "zero2one",
    normalize_scale: float = 1.0,
    score_lower_bound: float = 0.0,
    use_src_text: bool = True,
    use_ref_text: bool = True,
):
    """
    batch compute score
    """

    assert extra_infos is not None

    assert (
        isinstance(data_sources, Iterable)
        and isinstance(solution_strs, Iterable)
        and isinstance(ground_truths, Iterable)
        and isinstance(extra_infos, Iterable)
    )
    assert (
        len(data_sources)
        == len(solution_strs)
        == len(ground_truths)
        == len(extra_infos)
    )

    if use_extract_translation == "thinking":
        solution_strs = [extract_thinking_translation(s) for s in solution_strs]
    elif use_extract_translation == "no_thinking":
        solution_strs = [extract_no_thinking_translation(s) for s in solution_strs]
    elif use_extract_translation == "last_line":
        solution_strs = [extract_last_line(s) for s in solution_strs]
    elif use_extract_translation == "markdown":
        solution_strs = [extract_markdown(s) for s in solution_strs]
    elif use_extract_translation == "none":
        pass
    else:
        raise ValueError(f"Unknown use_extract_translation: {use_extract_translation}")

    # 提取辅助信息
    src_text = [extra_infos_item["src_text"] for extra_infos_item in extra_infos]
    tgt_text = [extra_infos_item["tgt_text"] for extra_infos_item in extra_infos]
    lg = [extra_infos_item["lang_pair"] for extra_infos_item in extra_infos]
    src_lang = [l.split("-")[0] for l in lg]
    trg_lang = [l.split("-")[1] for l in lg]

    extra_reward_info = [
        {
            "score": score_lower_bound,
            "mt_score": 0.0,
            "bleu_penalty": 0.0,
            "mt_length_penalty": 0.0,
            "long_resp_length_penalty": 0.0,
            "short_resp_length_penalty": 0.0,
        }
        for _ in solution_strs
    ]

    # 筛选非None的索引和对应数据
    valid_indices = [i for i, s in enumerate(solution_strs) if s is not None]
    valid_src_lang = [src_lang[i] for i in valid_indices]
    valid_trg_lang = [trg_lang[i] for i in valid_indices]
    valid_src_text = [src_text[i] for i in valid_indices]
    valid_trg_text = [tgt_text[i] for i in valid_indices]
    valid_solution_strs = [solution_strs[i] for i in valid_indices]
    valid_extra_reward_info = [extra_reward_info[i] for i in valid_indices]

    if en_proxy_reward:
        assert "en_text" in extra_infos[0]
        en_text = [extra_infos_item["en_text"] for extra_infos_item in extra_infos]
        valid_en_text = [en_text[i] for i in valid_indices]
    else:
        valid_en_text = None

    if use_length_penalty:
        assert response_lengths is not None
        assert len(response_lengths) == len(solution_strs)
        valid_response_lengths = [response_lengths[i] for i in valid_indices]

    # 仅对非None条目获取分数
    if valid_solution_strs:
        scores = get_tranlation_scores(
            valid_solution_strs,
            valid_src_lang,
            valid_trg_lang,
            valid_src_text,
            valid_trg_text,
            server_url,
            batch_size,
            normalize_type=normalize_type,
            normalize_scale=normalize_scale,
            use_src_text=use_src_text,
            use_ref_text=use_ref_text,
            texts_en=valid_en_text,
            use_bleu_penalty=use_bleu_penalty,
            use_length_penalty=use_length_penalty,
            extra_reward_info=valid_extra_reward_info,
        )
        scores = apply_response_length_penalty(
            scores,
            valid_response_lengths,
            max_response_len=filter_max_len,
            penalty_buffer_len=penalty_buffer_len,
            clip_score=score_lower_bound,
            min_response_len=short_penalty_buffer_len,
            extra_reward_info=valid_extra_reward_info,
        )

        scores = lower_bound_clip(scores, score_lower_bound)

        for idx, score in zip(valid_indices, scores):
            extra_reward_info[idx]["score"] = score

    return extra_reward_info


def extract_translation_progressive(solution_strs: str) -> tuple[str, str]:
    """Extracts the final answer from the model's response string.

    Args:
        solution_strs: Raw response string from the language model

    Returns:
        extracted_answer
    """
    forbidden_tags = [
        "<draft>",
        "</draft>",
        "<analysis>",
        "</analysis>",
    ] + TRANSLATION_TAGS
    # Extract final answer using XML-style tags
    draft_pattern = r"<draft>(.*?)</draft>"
    answer_pattern = TRANSLATION_REGEX_STR
    draft_matches = list(re.finditer(draft_pattern, solution_strs, re.DOTALL))

    if not draft_matches:
        draft_text = None
    else:
        draft_text = draft_matches[-1].group(1).strip()
        if not format_checker(draft_text, forbidden_tags):
            draft_text = None

    answer_matches = list(re.finditer(answer_pattern, solution_strs, re.DOTALL))

    if not answer_matches:
        answer_text = None
    else:
        answer_text = answer_matches[-1].group(1).strip()
        if not format_checker(answer_text, forbidden_tags):
            answer_text = None

    return draft_text, answer_text


def extract_translation_flexible_progressive(solution_strs: str) -> tuple[str, str]:
    """Extracts the final answer from the model's response string.

    Args:
        solution_strs: Raw response string from the language model

    Returns:
        extracted_answer
    """
    def search_translation(text:str, prefix_list: list[str]):
        text = text.strip()
        for prefix in prefix_list:
            if text.startswith(prefix):
                return text[len(prefix):].strip()
        return None

    solution_lines = solution_strs.strip().split("\n")
    if len(solution_lines) <= 3: # draft, analysis, answer
        return None, None

    draft_text, answer_text = solution_lines[0], solution_lines[-1]    
    draft_text = search_translation(draft_text, ["Draft translation:", "Draft Translation:"])
    answer_text = search_translation(answer_text, ["Final translation:", "Final Translation:"])
    if draft_text is None or answer_text is None:
        return None, None
    return draft_text, answer_text


def delta_plus_score(draft_score, answer_score, gamma=1.0):
    """
    $\frac{s_a-s_d}{1.1-s_d}+\gamma * s_d$
    The first term aims to reward improve, and the second term aims to reward draft quality, to prevent draft degradation.
    """
    draft_score = score_normalize(draft_score, 0, 1)
    answer_score = score_normalize(answer_score, 0, 1)

    delta = answer_score - draft_score
    return delta / (1.1 - draft_score) + gamma * draft_score


def delta_plus_plus_score(draft_score, answer_score):
    """
    $\frac{s_a-s_d}{1.1-s_d}*s_d$
    Combining the improve reward and draft reward together by multiplying.
    """
    draft_score = score_normalize(draft_score, 0, 1)
    answer_score = score_normalize(answer_score, 0, 1)

    delta = answer_score - draft_score
    return delta / (1.1 - draft_score) * draft_score


def sum_score(draft_score, answer_score):
    draft_score = score_normalize(draft_score, 0, 1)
    answer_score = score_normalize(answer_score, 0, 1)

    return draft_score + answer_score


def final_score(draft_score, answer_score):
    # draft_score = score_normalize(draft_score, 0, 1)
    answer_score = score_normalize(answer_score, 0, 1)
    return answer_score


def compute_score_progressive(
    data_sources,
    solution_strs,
    ground_truths,
    extra_infos=None,
    batch_size=128,
    fusion_type: str = "delta+",
    server_url: str = None,
    use_extract_translation: str = "break_line",
    use_bleu_penalty: bool = False,
    use_length_penalty: bool = False,
    filter_max_len: int = 1024,
    penalty_buffer_len: int = 256,
    response_lengths: list[int] = None,
    en_proxy_reward: bool = False,
    normalize_type: Literal["zero2one", "scale", "none"] = "zero2one",
    normalize_scale: float = 1.0,
    score_lower_bound: float = 0.0,
    use_src_text: bool = True,
    use_ref_text: bool = True,
):
    """
    batch compute score (progressive version aligned with compute_score)
    """

    assert fusion_type in ["delta+", "delta++", "sum", "final"]
    assert use_extract_translation  # 保持仅支持flexible_progressive提取

    if fusion_type == "delta+":
        fusion_fn = delta_plus_score
    elif fusion_type == "delta++":
        fusion_fn = delta_plus_plus_score
    elif fusion_type == "sum":
        fusion_fn = sum_score
    elif fusion_type == "final":
        fusion_fn = final_score

    assert extra_infos is not None
    assert (
        isinstance(data_sources, Iterable)
        and isinstance(solution_strs, Iterable)
        and isinstance(ground_truths, Iterable)
        and isinstance(extra_infos, Iterable)
    )
    assert (
        len(data_sources)
        == len(solution_strs)
        == len(ground_truths)
        == len(extra_infos)
    )

    # 提取翻译结果（草稿+最终）
    solution_strs_tuples = [extract_translation_flexible_progressive(s) for s in solution_strs]

    # 提取辅助信息
    src_text = [extra_infos_item["src_text"] for extra_infos_item in extra_infos]
    tgt_text = [extra_infos_item["tgt_text"] for extra_infos_item in extra_infos]
    lg = [extra_infos_item["lang_pair"] for extra_infos_item in extra_infos]
    src_lang = [l.split("-")[0] for l in lg]
    trg_lang = [l.split("-")[1] for l in lg]

    # 初始化分数为下界值
    final_scores = [score_lower_bound] * len(solution_strs_tuples)

    # 筛选有效索引（草稿和最终翻译均非None）
    valid_indices = []
    for i, t in enumerate(solution_strs_tuples):
        if t[0] is not None and t[1] is not None:
            valid_indices.append(i)

    valid_src_lang = [src_lang[i] for i in valid_indices]
    valid_trg_lang = [trg_lang[i] for i in valid_indices]
    valid_src_text = [src_text[i] for i in valid_indices]
    valid_trg_text = [tgt_text[i] for i in valid_indices]
    valid_draft_strs = [solution_strs_tuples[i][0] for i in valid_indices]
    valid_answer_strs = [solution_strs_tuples[i][1] for i in valid_indices]
    
    # 提取英文代理文本（若启用）
    if en_proxy_reward:
        assert "en_text" in extra_infos[0]
        en_text = [extra_infos_item["en_text"] for extra_infos_item in extra_infos]
        valid_en_text = [en_text[i] for i in valid_indices]
    else:
        valid_en_text = None

    # 处理响应长度（分离草稿和最终长度）
    if use_length_penalty:
        assert response_lengths is not None
        assert len(response_lengths) == len(solution_strs)
        valid_response_lengths = [response_lengths[i] for i in valid_indices]


    # 仅处理有效条目
    if valid_draft_strs:
        # 辅助函数：处理分数计算流程
        

        # 计算草稿分数
        draft_scores = get_tranlation_scores(
            valid_draft_strs,
            valid_src_lang,
            valid_trg_lang,
            valid_src_text,
            valid_trg_text,
            server_url,
            batch_size,
            normalize_type=normalize_type,
            normalize_scale=normalize_scale,
            use_src_text=use_src_text,
            use_ref_text=use_ref_text,
            texts_en=valid_en_text,
            en_proxy_reward=en_proxy_reward,
            use_bleu_penalty=use_bleu_penalty,
            use_length_penalty=use_length_penalty,
        )
        
        # 计算最终答案分数
        answer_scores = get_tranlation_scores(
            valid_answer_strs,
            valid_src_lang,
            valid_trg_lang,
            valid_src_text,
            valid_trg_text,
            server_url,
            batch_size,
            normalize_type=normalize_type,
            normalize_scale=normalize_scale,
            use_src_text=use_src_text,
            use_ref_text=use_ref_text,
            texts_en=valid_en_text,
            en_proxy_reward=en_proxy_reward,
            use_bleu_penalty=use_bleu_penalty,
            use_length_penalty=use_length_penalty,
        )
        
        # 调试信息
        print(f"[Info] draft score avg: {sum(draft_scores) / len(draft_scores):.4f}")
        print(f"[Info] answer score avg: {sum(answer_scores) / len(answer_scores):.4f}")
        print(
            f"[Info] delta score avg: "
            f"{(sum(answer_scores) - sum(draft_scores)) / len(answer_scores):.4f}"
        )
        
        # 融合分数
        scores = [
            fusion_fn(d, a) 
            for d, a in zip(draft_scores, answer_scores)
        ]

        scores = apply_response_length_penalty(
            scores,
            valid_response_lengths,
            max_response_len=filter_max_len,
            penalty_buffer_len=penalty_buffer_len,
            clip_score=score_lower_bound,
        )
        
        # 更新有效索引的分数
        for idx, score in zip(valid_indices, scores):
            final_scores[idx] = score

    return final_scores


def get_tranlation_scores(
    texts_mt, 
    langs_src, 
    langs_trg, 
    texts_src, 
    texts_trg,
    server_url,
    batch_size,
    normalize_type,
    normalize_scale,
    use_src_text = True,
    use_ref_text = True,
    texts_en = None,
    en_proxy_reward = False,
    use_bleu_penalty = False,
    use_length_penalty = False,
    extra_reward_info = None,
):
    """统一处理分数计算流程"""
    # 获取原始分数
    if en_proxy_reward:
        assert texts_en is not None
        raw_scores = get_rewards_from_server(
            src_langs=["en"] * len(texts_mt),
            trg_langs=langs_trg,
            response_texts=texts_mt,
            server_url=server_url,
            batch_size=batch_size,
            src_texts=texts_en if use_src_text else None,
            ref_texts=texts_trg if use_ref_text else None,
        )
    else:
        raw_scores = get_rewards_from_server(
            src_langs=langs_src,
            trg_langs=langs_trg,
            response_texts=texts_mt,
            server_url=server_url,
            batch_size=batch_size,
            src_texts=texts_src if use_src_text else None,
            ref_texts=texts_trg if use_ref_text else None,
        )

    if extra_reward_info is not None:
        for i, raw_score in enumerate(raw_scores):
            extra_reward_info[i]["mt_score"] = raw_score
    
    # 分数归一化
    if normalize_type == "zero2one":
        scores = [score_normalize(s, 0, 1) for s in raw_scores]
    elif normalize_type == "scale":
        scores = [s * normalize_scale for s in raw_scores]
    elif normalize_type == "none":
        scores = raw_scores
    else:
        raise ValueError(f"Invalid normalize_type: {normalize_type}")
    
    
    # BLEU惩罚
    if use_bleu_penalty:
        penalties = get_bleu_penalty(
            texts_mt,
            texts_src,
            texts_trg,
            langs_src,
        )
        scores = [s - p for s, p in zip(scores, penalties)]
        if extra_reward_info is not None:
            for i, penalty in enumerate(penalties):
                extra_reward_info[i]["bleu_penalty"] = penalty
    
    # 长度惩罚
    if use_length_penalty:
        scores = apply_length_penalty(
            scores,
            texts_mt,
            texts_trg,
        )
        if extra_reward_info is not None:
            for i, score in enumerate(scores):
                extra_reward_info[i]["mt_length_penalty"] = score
    
    return scores


def replace_numbers_and_operators(input_string):
    # 定义正则表达式模式，匹配数字和数学运算符
    pattern = r"[0-9+\-*/=()]"
    # 使用re.sub将匹配的字符替换为空格
    result = re.sub(pattern, "@", input_string)
    return result


PAD_STR = " [PxAxD]" * 50
PAD_STR = "{}" + PAD_STR


def add_padding(input_string):
    # Add padding to increase mt length, so we can get rid of BLEU length penalty
    return PAD_STR.format(input_string)


def compute_cross_bleu(mt_list: list, ref_list: list, trg_lang: str = "en"):
    import comet_reward.sacrebleu_eval as sacrebleu_eval

    eval_out = sacrebleu_eval.func_call(
        mt_list,
        ref_list,
        trg_lang=trg_lang,
        lowercase=True,
    )
    assert len(mt_list) == len(eval_out["scores"])
    return eval_out["scores"]


def get_bleu_penalty(
    mt_list: list,
    src_list: list,
    ref_list: list,
    src_langs: list,
    ref_lang: str = "en",
    scale: float = 0.2,
):
    mt_list = [add_padding(replace_numbers_and_operators(mt)) for mt in mt_list]

    # 分组处理不同语言的src_cross_bleu
    zh_indices = [i for i, lang in enumerate(src_langs) if lang == "zh"]
    non_zh_indices = [i for i, lang in enumerate(src_langs) if lang != "zh"]

    # 初始化结果列表
    src_cross_bleu = [0] * len(mt_list)

    # 处理中文组
    if zh_indices:
        zh_mt = [mt_list[i] for i in zh_indices]
        zh_src = [src_list[i] for i in zh_indices]
        zh_scores = compute_cross_bleu(zh_mt, zh_src, trg_lang="zh")
        for idx, score in zip(zh_indices, zh_scores):
            src_cross_bleu[idx] = score

    # 处理非中文组
    if non_zh_indices:
        non_zh_mt = [mt_list[i] for i in non_zh_indices]
        non_zh_src = [src_list[i] for i in non_zh_indices]
        non_zh_scores = compute_cross_bleu(non_zh_mt, non_zh_src, trg_lang="en")
        for idx, score in zip(non_zh_indices, non_zh_scores):
            src_cross_bleu[idx] = score

    # 计算参考BLEU (始终使用en)
    ref_cross_bleu = compute_cross_bleu(mt_list, ref_list, trg_lang=ref_lang)

    # 计算最终惩罚
    bleu_penalty = [(s + r) * scale for s, r in zip(src_cross_bleu, ref_cross_bleu)]

    print(
        "[Info] bleu penalty avg - {:.3f}  min - {:.3f}  max - {:.3f}".format(
            sum(bleu_penalty) / len(bleu_penalty), min(bleu_penalty), max(bleu_penalty)
        )
    )

    return bleu_penalty


def compute_length_penalty(length: int, min_length: int, max_length: int):
    """
    length penalty from 0 to 1 (0: min_length, 1: max_length)
    """
    penalty = (length - min_length) / (max_length - min_length)
    return score_normalize(penalty, 0, 1)


def apply_length_penalty(
    scores: list,
    mt_list: list,
    ref_list: list,
    min_length_factor: float = 2,
    max_length_factor: float = 4,
):
    length_penalty = [
        compute_length_penalty(
            len(mt), len(ref) * min_length_factor, len(ref) * max_length_factor
        )
        for mt, ref in zip(mt_list, ref_list)
    ]

    print(
        "[Info] length penalty avg - {:.3f}  min - {:.3f}  max - {:.3f}".format(
            sum(length_penalty) / len(length_penalty),
            min(length_penalty),
            max(length_penalty),
        )
    )

    scores = [s - p for s, p in zip(scores, length_penalty)]
    return scores


def apply_response_length_penalty(
    scores: list,
    response_token_len: list,
    max_response_len: int,
    penalty_buffer_len: int,
    clip_score: float = 0.0, # score for response len >= max_response_len
    min_response_len: int = 0, # min response len
    extra_reward_info: list[dict] = None,
):
    """
    Penalty based on response len.
    The penalty increases from 0 to 1 as the response length changes from max_response_len-penalty_buffer_len to max_response_len.
    For response length exceeds (or equal to) max_response_len, the penalty is clip_score.

    If min_response_len > 0, the penalty will be applied to response length less than min_response_len from 0 to 1 as length changes from min_response_len to 0.
    """
    if penalty_buffer_len > 0:
        length_penalty = [
            compute_length_penalty(
                length_item, max_response_len-penalty_buffer_len, max_response_len
            )
            for length_item in response_token_len
        ]

        if extra_reward_info is not None:
            for i in range(len(scores)):
                extra_reward_info[i]["long_resp_length_penalty"] = length_penalty[i]
        
    scores = [s - p for s, p in zip(scores, length_penalty)]

    if min_response_len > 0:
        length_penalty = [
            1-compute_length_penalty(
                length_item, 0, min_response_len
            )
            for length_item in response_token_len
        ]
        scores = [s - p for s, p in zip(scores, length_penalty)]

        if extra_reward_info is not None:
            for i in range(len(scores)):
                extra_reward_info[i]["short_resp_length_penalty"] = length_penalty[i]

    filtered_count = 0
    for i in range(len(response_token_len)):
        if response_token_len[i] >= max_response_len:
            scores[i] = clip_score
            filtered_count += 1
    print(
        "[Info] response length penalty avg - {:.3f}  min - {:.3f}  max - {:.3f} - filtered - {} / {}".format(
            sum(length_penalty) / len(length_penalty),
            min(length_penalty),
            max(length_penalty),
            filtered_count,
            len(scores),
        )
    )
    return scores


def lower_bound_clip(scores: list, lower_bound_score: float = 0.0):
    """
    Clip scores to lower_bound_score.
    """
    for i in range(len(scores)):
        if scores[i] < lower_bound_score:
            scores[i] = lower_bound_score
    return scores

from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Iterable
import requests
import time
import re

SERVER_URL = "http://[2605:340:cd51:601:953b:de9f:af10:b2bb]:9658/predict"


def score_normalize(score: float, min_score: float, max_score: float) -> float:
    """
    clip score
    """
    return min(max(score, min_score), max_score)


import time
from typing import List, Literal
import requests


def get_rewards_from_server(
    src_langs: List[str],
    trg_langs: List[str],
    ref_langs: List[str],
    src_texts: List[str],
    ref_texts: List[str],
    response_texts: List[str],
    rm_type: Literal["direct", "pivot", "msr"] = "direct",
    server_url: str = None,
    batch_size: int = 128,
) -> List[float]:
    r"""
    Gets reward scores from the API server by splitting requests into batches.
    """
    if server_url is None:
        server_url = SERVER_URL

    total_items = len(src_texts)
    all_rewards = []

    # 处理单个批次的请求（包含重试逻辑）
    def _request_single_batch(batch_indices):
        batch_payload = {
            "src_list": [src_texts[i] for i in batch_indices],
            "mt_list": [response_texts[i] for i in batch_indices],
            "src_langs": [src_langs[i] for i in batch_indices],
            "trg_langs": [trg_langs[i] for i in batch_indices],
            "ref_list": [ref_texts[i] for i in batch_indices],
            "ref_langs": [ref_langs[i] for i in batch_indices],
            "rm_type": rm_type,
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


def format_checker(solution_strs: str, forbidden_tags: list[str]) -> bool:
    for tag in forbidden_tags:
        if tag in solution_strs:
            return False
    if solution_strs == "":
        return False
    return True


def extract_translation(solution_strs: str):
    """Extracts the final answer from the model's response string.

    Args:
        solution_strs: Raw response string from the language model

    Returns:
        extracted_answer
    """
    # Extract final answer using XML-style tags
    answer_pattern = r"<translate>(.*?)</translate>"
    matches = list(re.finditer(answer_pattern, solution_strs, re.DOTALL))

    if not matches:
        # print("[Error] No valid answer tags found")
        return None

    final_answer = matches[-1].group(1).strip()

    if format_checker(
        final_answer, ["<think>", "<translate>", "</think>", "</translate>"]
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
    use_extract_translation: bool = True,
    use_bleu_penalty: bool = False,
    use_length_penalty: bool = False,
    filter_max_len: int = 1024,
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

    # 提取翻译结果，可能包含None
    if use_extract_translation:
        solution_strs = [extract_translation(s) for s in solution_strs]

    # 提取辅助信息
    src_text = [extra_infos_item["src_text"] for extra_infos_item in extra_infos]
    tgt_text = [extra_infos_item["tgt_text"] for extra_infos_item in extra_infos]
    lg = [extra_infos_item["lang_pair"] for extra_infos_item in extra_infos]
    src_lang = [l.split("-")[0] for l in lg]
    trg_lang = [l.split("-")[1] for l in lg]

    # 初始化全0分数列表
    scores = [0] * len(solution_strs)

    # 筛选非None的索引和对应数据
    non_none_indices = [i for i, s in enumerate(solution_strs) if s is not None]
    non_none_src_lang = [src_lang[i] for i in non_none_indices]
    non_none_trg_lang = [trg_lang[i] for i in non_none_indices]
    non_none_src_text = [src_text[i] for i in non_none_indices]
    non_none_trg_text = [tgt_text[i] for i in non_none_indices]
    non_none_solution_strs = [solution_strs[i] for i in non_none_indices]

    # 仅对非None条目获取分数
    if non_none_solution_strs:
        non_none_scores = get_rewards_from_server(
            src_langs=non_none_src_lang,
            trg_langs=non_none_trg_lang,
            ref_langs=non_none_trg_lang,
            src_texts=non_none_src_text,
            ref_texts=non_none_trg_text,
            response_texts=non_none_solution_strs,
            rm_type="direct",
            server_url=server_url,
            batch_size=batch_size,
        )

        # 归一化分数并更新到对应位置
        normalized_scores = [score_normalize(s, 0, 1) for s in non_none_scores]
        if use_bleu_penalty:
            penalty_scores = get_bleu_penalty(
                non_none_solution_strs,
                non_none_src_text,
                non_none_trg_text,
                non_none_src_lang,
            )
            normalized_scores = [
                s - p for s, p in zip(normalized_scores, penalty_scores)
            ]

        if use_length_penalty:
            response_lengths = [
                extra_infos[i]["response_length"] for i in non_none_indices
            ]
            assert len(response_lengths) == len(normalized_scores)

            normalized_scores = apply_length_penalty_filter(
                normalized_scores,
                non_none_solution_strs,
                non_none_trg_text,
                response_lengths,
                max_response_len=filter_max_len,
            )

        for idx, score in zip(non_none_indices, normalized_scores):
            scores[idx] = score

    return scores


def extract_translation_progressive(solution_strs: str) -> tuple[str, str]:
    """Extracts the final answer from the model's response string.

    Args:
        solution_strs: Raw response string from the language model

    Returns:
        extracted_answer
    """
    forbidden_tags = [
        "<draft>",
        "<translation>",
        "</draft>",
        "</translation>",
        "<analysis>",
        "</analysis>",
    ]
    # Extract final answer using XML-style tags
    draft_pattern = r"<draft>(.*?)</draft>"
    answer_pattern = r"<translation>(.*?)</translation>"
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
    use_extract_translation: bool = True,
):
    """
    batch compute score
    """

    assert fusion_type in ["delta+", "delta++", "sum", "final"]
    assert use_extract_translation

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

    # 提取翻译结果，可能包含None
    solution_strs_tuples = [extract_translation_progressive(s) for s in solution_strs]

    # 提取辅助信息
    src_text = [extra_infos_item["src_text"] for extra_infos_item in extra_infos]
    tgt_text = [extra_infos_item["tgt_text"] for extra_infos_item in extra_infos]
    lg = [extra_infos_item["lang_pair"] for extra_infos_item in extra_infos]
    src_lang = [l.split("-")[0] for l in lg]
    trg_lang = [l.split("-")[1] for l in lg]

    # 初始化全0分数列表
    scores = [0] * len(solution_strs_tuples)

    non_none_indices = []
    # 筛选非None的索引和对应数据
    for i, t in enumerate(solution_strs_tuples):
        # t[0]: draft; t[1]: answer
        if t[0] is not None and t[1] is not None:
            non_none_indices.append(i)

    non_none_src_lang = [src_lang[i] for i in non_none_indices]
    non_none_trg_lang = [trg_lang[i] for i in non_none_indices]
    non_none_src_text = [src_text[i] for i in non_none_indices]
    non_none_trg_text = [tgt_text[i] for i in non_none_indices]
    non_none_draft_strs = [solution_strs_tuples[i][0] for i in non_none_indices]
    non_none_answer_strs = [solution_strs_tuples[i][1] for i in non_none_indices]

    # 仅对非None条目获取分数
    if non_none_draft_strs:
        non_none_draft_scores = get_rewards_from_server(
            src_langs=non_none_src_lang,
            trg_langs=non_none_trg_lang,
            ref_langs=non_none_trg_lang,
            src_texts=non_none_src_text,
            ref_texts=non_none_trg_text,
            response_texts=non_none_draft_strs,
            rm_type="direct",
            server_url=server_url,
            batch_size=batch_size,
        )

        non_none_answer_scores = get_rewards_from_server(
            src_langs=non_none_src_lang,
            trg_langs=non_none_trg_lang,
            ref_langs=non_none_trg_lang,
            src_texts=non_none_src_text,
            ref_texts=non_none_trg_text,
            response_texts=non_none_answer_strs,
            rm_type="direct",
            server_url=server_url,
            batch_size=batch_size,
        )

        print(
            "[Info] draft score avg: ",
            sum(non_none_draft_scores) / len(non_none_draft_scores),
        )
        print(
            "[Info] answer score avg: ",
            sum(non_none_answer_scores) / len(non_none_answer_scores),
        )
        print(
            "[Info] delta score avg: ",
            (sum(non_none_answer_scores) - sum(non_none_draft_scores))
            / len(non_none_answer_scores),
        )

        final_score = [
            fusion_fn(d, a)
            for d, a in zip(non_none_draft_scores, non_none_answer_scores)
        ]

        for idx, score in zip(non_none_indices, final_score):
            scores[idx] = score

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


def get_length_penalty(
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
    return length_penalty


def apply_length_penalty_filter(
    scores: list,
    mt_list: list,
    ref_list: list,
    response_token_len: list,
    max_response_len: int,
    filtered_score: float = -2,
):

    penalty_scores = get_length_penalty(mt_list, ref_list)
    scores = [s - p for s, p in zip(scores, penalty_scores)]

    filter_count = 0
    for i in range(len(response_token_len)):  # overwrite scores
        if response_token_len[i] >= max_response_len:
            scores[i] = filtered_score
            filter_count += 1
    print("[Info] length filtered sample: {}".format(filter_count))
    return scores

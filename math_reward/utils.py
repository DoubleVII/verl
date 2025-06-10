import re

def format_checker(solution_strs: str, forbidden_tags: list[str]) -> bool:
    for tag in forbidden_tags:
        if tag in solution_strs:
            return False
    if solution_strs == "":
        return False
    return True

def presence_checker(solution_strs: str, presence_tags: list[str], force_once: bool = False) -> bool:
    for tag in presence_tags:
        presence_count = solution_strs.count(tag)
        if presence_count == 0:
            return False
        if presence_count > 1 and force_once:
            return False
    return True

def extract_solution(solution_strs: str):
    """Extracts the final answer from the model's response string.

    Args:
        solution_strs: Raw response string from the language model

    Returns:
        extracted_answer
    """
    presence_tags = ["<think>", "<answer>", "</think>", "</answer>"]
    if not presence_checker(solution_strs, presence_tags, force_once=True):
        return None
    # Extract final answer using XML-style tags
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_strs, re.DOTALL))

    if not matches:
        # print("[Error] No valid answer tags found")
        return None

    final_answer = matches[-1].group(1).strip()

    if format_checker(
        final_answer, ["<think>", "<answer>", "</think>", "</answer>"]
    ):
        return final_answer
    return None


def compute_score(data_source, solution_str, ground_truth, extra_info=None, format_score=0.0, score=1.0, use_extract_solution: bool = True,):
    if use_extract_solution:
        answer = extract_solution(solution_str=solution_str)
    else:
        answer = solution_str.strip()
    
    if answer is None:
        return 0
    else:
        if answer == ground_truth:
            return score
        else:
            return format_score

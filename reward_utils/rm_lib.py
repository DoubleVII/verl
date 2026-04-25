from typing import Optional, List, Dict, Tuple, Iterable

try:
    from .language_detector import is_language_match
except ImportError:
    from reward_utils.language_detector import is_language_match

try:
    from .post_edit_reward_remote import get_GQM_with_notes_prompt
except ImportError:
    from reward_utils.post_edit_reward_remote import get_GQM_with_notes_prompt


def _line_extractor(response: str) -> Optional[str]:
    response = response.strip()
    if not response:
        return None
    last = response.split("\n")[-1].strip()
    if not last:
        return None
    return last

def _block_extractor(response: str) -> Optional[str]:
    response = response.strip()
    if response.count("```") != 2:
        return None
    if not response:
        return None
    if not response.endswith("```"):
        return None
    response = response[:-3]
    block_start = response.rfind("```")
    if block_start == -1:
        return None
    extract_out = response[block_start + 3 :].strip()
    if not extract_out:
        return None
    return extract_out


def _one_line_extractor(response: str) -> Optional[str]:
    response = response.strip()
    if not response:
        return None
    if "\n" in response:
        return None
    return response
    

def _decode_response(data, src_tokenizer, extractor_type: str = "line") -> List[Optional[str]]:
    response_list: List[Optional[str]] = []

    for i in range(data.batch.batch_size[0]):
        # extract response
        response_ids = data.batch["responses"][i]
        response_length = response_ids.shape[-1]
        valid_response_length = data.batch["attention_mask"][i][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # decode
        response = src_tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        response = response.replace(src_tokenizer.eos_token, "")
        if extractor_type == "line":
            extracted = _line_extractor(response)
        elif extractor_type == "codeblock":
            extracted = _block_extractor(response)
        elif extractor_type == "oneline":
            extracted = _one_line_extractor(response)
        elif extractor_type == "none":
            extracted = response.strip()
        else:
            raise ValueError(f"extractor_type: {extractor_type}")

        response_list.append(extracted)

    return response_list


def _get_lang_pair(extra_info: dict) -> tuple:
    """
    extra_info: str, e.g. "en-zh"
    """
    if "src_lang" in extra_info and "trg_lang" in extra_info:
        src_lang = extra_info["src_lang"]
        tgt_lang = extra_info["trg_lang"]
    elif "lang_pair" in extra_info:
        src_lang, tgt_lang = extra_info["lang_pair"].split("-")
    else:
        raise ValueError(f"extra_info: {extra_info}")
    return src_lang, tgt_lang

LANG_MAP = {
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "ru": "Russian",
    "ko": "Korean",
    "ja": "Japanese",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "it": "Italian",
    "nl": "Dutch",
    "th": "Thai",
    "ro": "Romanian",
    "ar": "Arabic",
    "el": "Greek",
    "vi": "Vietnamese",
}


promt_template = """Given a source text in {} and a translation text in {}. Perform a step by step analysis of translation quality and assign a score on a scale from 0 to 10.
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
    return promt_template.format(src_lang, tgt_lang, src_text, mt_text)

def single_extract_score(output_text: str) -> Optional[float]:
    output_text = output_text.strip()
    try:
        last_line_index = output_text.rfind("\n")
        last_line = output_text[last_line_index:].strip()
        score = int(last_line)
        return float(score)
    except Exception:
        return None


CANDIDATE_IDENTIFIERS = [
    "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"
]

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
    if len(mt_texts) > len(CANDIDATE_IDENTIFIERS):
        raise ValueError(f"Only support {len(CANDIDATE_IDENTIFIERS)} candidates.")
    task_prompt = _group_get_task_prompt(prompt_format, add_example)
    candidate_prompts = "".join([CANDIDATE_PROMPT.format(CANDIDATE_IDENTIFIERS[i], mt_texts[i]) for i in range(len(mt_texts))])
    return GROUP_PROMPT_TEMPLATE.format(source_lang, target_lang, task_prompt, source_text, candidate_prompts)

def _group_validate_ranking(text: str, expected_num: int) -> bool:
    try:
        if "<" in text:
            return False
        tiers = []
        for group in text.split('>'):
            tier = set(x.strip() for x in group.split('='))
            tiers.append(tier)
        count = sum(len(t) for t in tiers)
        if count != expected_num:
            return False
        for cid in CANDIDATE_IDENTIFIERS[:expected_num]:
            if text.count(cid) != 1:
                return False
        return True
    except Exception:
        return False

def _group_ranking_to_scores(ranking_text: str) -> Dict[str, int]:
    tiers = ranking_text.split('>')
    score_map = {}
    max_score = len(tiers) - 1
    for i, tier in enumerate(tiers):
        for cid in tier.split('='):
            score_map[cid.strip()] = max_score - i
    return score_map

def _group_parse_score_text(score_text: str) -> Optional[Dict[str, int]]:
    try:
        result = {}
        for it in score_text.split(','):
            k, v = it.split(':')
            result[k.strip()] = int(v.strip())
        return result
    except Exception:
        return None

def group_extract_scores(output_text: str, prompt_type: str, expected_num: int) -> Optional[List[int]]:
    output_text = output_text.strip()
    try:
        if "\n" not in output_text:
            last_line = output_text
        else:
            idx = output_text.rfind("\n")
            last_line = output_text[idx:].strip()
        if prompt_type == "score":
            scores = [int(s.strip().split(":")[-1]) for s in last_line.split(",")]
            if len(scores) != expected_num:
                return None
            return scores
        if prompt_type == "ranking":
            if not _group_validate_ranking(last_line, expected_num):
                return None
            score_map = _group_ranking_to_scores(last_line)
            scores = [score_map[cid] for cid in CANDIDATE_IDENTIFIERS[:expected_num]]
            return scores
        if prompt_type == "ranking_score":
            score_dict = _group_parse_score_text(last_line)
            if score_dict is None:
                return None
            scores = [score_dict[cid] for cid in CANDIDATE_IDENTIFIERS[:expected_num]]
            return scores
        raise ValueError(f"prompt_type must be one of {GROUP_TASK_FORMAT.keys()}")
    except Exception:
        return None

def _compute_overlong_penalty(length: int, overlong_buffer_cfg: Optional[Dict]) -> float:
    if not overlong_buffer_cfg:
        return 0.0
    if not overlong_buffer_cfg.get("enable", False):
        return 0.0
    max_resp_len = overlong_buffer_cfg.get("max_resp_len", None)
    buffer_len = overlong_buffer_cfg.get("len", 0)
    penalty_factor = overlong_buffer_cfg.get("penalty_factor", 0.0)
    if max_resp_len is None or buffer_len <= 0 or penalty_factor <= 0.0:
        return 0.0
    threshold = max_resp_len - buffer_len
    if length <= threshold:
        return 0.0
    delta = length - threshold
    if delta >= buffer_len:
        return penalty_factor
    return penalty_factor * (delta / buffer_len)

class RewardModelProcessor:
    def __init__(self, *args, **kwargs):
        self.config = kwargs.get("config")
        self.tokenizer = kwargs.get("tokenizer", None)
        self.input_tokenizer = kwargs.get("input_tokenizer", self.tokenizer)
        self.max_prompt_length = self.config.prompt_length
        self.extractor_type = self.config.custom_processor.get("extractor_type", "line")
        print(f"Using extractor_type: {self.extractor_type}")
        self.score_scale_factor = getattr(self.config, "score_scale_factor", 0.1)
        self.default_reward = getattr(self.config, "default_reward", 0.0)
        self.overlong_buffer_cfg = self.config.custom_processor.get("overlong_buffer", None)
        self.enable_language_detection = self.config.custom_processor.get("enable_language_detection", False)
        if self.enable_language_detection:
            print(f"Language detection enabled")
        if self.tokenizer is None:
            raise ValueError("tokenizer must be provided")
        if self.input_tokenizer is None:
            raise ValueError("input_tokenizer must be provided")

    def process_input(self, data):
        response_list = _decode_response(data, self.input_tokenizer, self.extractor_type)
        src_text_list = [item["src_text"] for item in data.non_tensor_batch["extra_info"]]
        lang_pair_list = [_get_lang_pair(item) for item in data.non_tensor_batch["extra_info"]]
        src_langs, tgt_langs = zip(*lang_pair_list)
        
        assert len(src_text_list) == len(response_list) == len(src_langs) == len(tgt_langs)
        prompt_list = []
        kept_indices = []
        filtered_indices = []
        for idx, (src_text, mt_text, src_lang, tgt_lang) in enumerate(zip(src_text_list, response_list, src_langs, tgt_langs)):
            if mt_text is None:
                filtered_indices.append(idx)
                continue
            if self.enable_language_detection:
                if not is_language_match(mt_text, tgt_lang):
                    filtered_indices.append(idx)
                    continue
            prompt = single_get_prompt(src_text, mt_text, src_lang, tgt_lang)
            messages = [
                {"role": "user", "content": prompt},
            ]
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            raw_prompt_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
            # filter by max_prompt_length; too-long entries will be scored as 0 later
            if len(raw_prompt_ids) > self.max_prompt_length:
                filtered_indices.append(idx)
                continue
            kept_indices.append(idx)
            prompt_list.append({"prompt_token_ids": raw_prompt_ids})
        total_size = len(src_text_list)
        return prompt_list, kept_indices, total_size
    
    def process_output(self, outputs, data, kept_indices, total_size) -> List[float]:
        final_scores: list[float] = [self.default_reward] * total_size
        for j, output in enumerate(outputs):
            text = output.outputs[0].text
            score = single_extract_score(text)
            if score is None:
                score = self.default_reward
            else:
                score = score * self.score_scale_factor
            if j < len(kept_indices):
                idx = kept_indices[j]
                response_ids = data.batch["responses"][idx]
                resp_len = response_ids.shape[-1]
                valid_len = data.batch["attention_mask"][idx][-resp_len:].sum()
                try:
                    valid_len_int = int(valid_len)
                except Exception:
                    valid_len_int = resp_len
                penalty = _compute_overlong_penalty(valid_len_int, self.overlong_buffer_cfg)
                score = score - penalty
                final_scores[kept_indices[j]] = score

        # filtered indices remain default_reward
        return final_scores

    def compute_scores(self, data, generate_fn):
        prompt_list, kept_indices, total_size = self.process_input(data)
        outputs = generate_fn(prompt_list)
        return self.process_output(outputs, data, kept_indices, total_size)


class GroupRewardModelProcessor:
    def __init__(self, *args, **kwargs):
        self.config = kwargs.get("config")
        self.tokenizer = kwargs.get("tokenizer", None)
        self.input_tokenizer = kwargs.get("input_tokenizer", self.tokenizer)
        self.max_prompt_length = self.config.prompt_length
        self.extractor_type = self.config.custom_processor.get("extractor_type", "line")
        print(f"Using extractor_type: {self.extractor_type}")
        self.prompt_type = getattr(self.config, "group_prompt_type", "ranking_score")
        self.add_example = getattr(self.config, "group_add_example", False)
        self.score_scale_factor = getattr(self.config, "score_scale_factor", 0.1)
        self.default_reward = getattr(self.config, "default_reward", 0.0)
        self.overlong_buffer_cfg = self.config.custom_processor.get("overlong_buffer", None)
        self.enable_language_detection = self.config.custom_processor.get("enable_language_detection", False)
        if self.enable_language_detection:
            print(f"Language detection enabled")
        if self.tokenizer is None:
            raise ValueError("tokenizer must be provided")
        if self.input_tokenizer is None:
            raise ValueError("input_tokenizer must be provided")

    def _group_indices(self, uids: List):
        groups: Dict[str, List[int]] = {}
        for idx, uid in enumerate(uids):
            key = str(uid)
            if key not in groups:
                groups[key] = []
            groups[key].append(idx)
        return list(groups.items())

    def process_input(self, data) -> Tuple[List[Dict[str, List[int]]], List[Dict[str, List[List[int]]]], List[List[int]], int]:
        responses = _decode_response(data, self.input_tokenizer, self.extractor_type)
        extra = data.non_tensor_batch["extra_info"]
        uids = data.non_tensor_batch.get("uid", None)
        if uids is None:
            raise ValueError("uid not found in batch")
        groups = self._group_indices(list(uids))

        prompt_list: List[Dict[str, List[int]]] = []
        kept_groups: List[Dict[str, List[List[int]]]] = []
        zero_groups: List[List[int]] = []
        for uid_key, indices in groups:
            src_text = extra[indices[0]]["src_text"]
            src_lang, tgt_lang = _get_lang_pair(extra[indices[0]])
            if len(src_lang) == 2:
                src_lang = LANG_MAP[src_lang]
            if len(tgt_lang) == 2:
                tgt_lang = LANG_MAP[tgt_lang]
            seen: Dict[str, int] = {}
            unique_texts: List[str] = []
            dup_map: List[List[int]] = []
            invalid_indices: List[int] = []
            for idx in indices:
                t = responses[idx]
                if t is None:
                    invalid_indices.append(idx)
                    continue
                if self.enable_language_detection:
                    if not is_language_match(t, extra[indices[0]]["trg_lang"] if "trg_lang" in extra[indices[0]] else extra[indices[0]]["lang_pair"].split("-")[1]):
                        invalid_indices.append(idx)
                        continue
                if t in seen:
                    dup_map[seen[t]].append(idx)
                else:
                    seen[t] = len(unique_texts)
                    unique_texts.append(t)
                    dup_map.append([idx])
            valid_indices = [i for i in indices if i not in invalid_indices]
            if len(unique_texts) <= 1:
                if valid_indices:
                    zero_groups.append(valid_indices)
                for inv in invalid_indices:
                    zero_groups.append([inv])
                continue
            prompt = group_get_prompt(src_lang, tgt_lang, src_text, unique_texts, self.prompt_type, add_example=self.add_example)
            messages = [{"role": "user", "content": prompt}]
            input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            raw_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
            if len(raw_ids) > self.max_prompt_length:
                if valid_indices:
                    zero_groups.append(valid_indices)
                for inv in invalid_indices:
                    zero_groups.append([inv])
                continue
            candidate_lens: List[int] = []
            for targets in dup_map:
                first_idx = targets[0]
                response_ids = data.batch["responses"][first_idx]
                resp_len = response_ids.shape[-1]
                valid_len = data.batch["attention_mask"][first_idx][-resp_len:].sum()
                try:
                    candidate_lens.append(int(valid_len))
                except Exception:
                    candidate_lens.append(resp_len)
            kept_groups.append({"uid": [uid_key], "dup_map": dup_map, "candidate_lens": candidate_lens})
            prompt_list.append({"prompt_token_ids": raw_ids})
        total_size = len(responses)
        return prompt_list, kept_groups, zero_groups, total_size

    def process_output(self, outputs, kept_groups, zero_groups, total_size) -> List[float]:
        final_scores: List[float] = [self.default_reward] * total_size
        for j, output in enumerate(outputs):
            text = output.outputs[0].text
            group_info = kept_groups[j]
            dup_map = group_info["dup_map"]
            candidate_lens = group_info.get("candidate_lens", [0] * len(dup_map))
            scores = group_extract_scores(text, self.prompt_type, len(dup_map))
            if scores is None:
                scores = [self.default_reward] * len(dup_map)
            normalized = [s * self.score_scale_factor for s in scores]
            for k, targets in enumerate(dup_map):
                penalty = _compute_overlong_penalty(candidate_lens[k], self.overlong_buffer_cfg)
                sc = normalized[k] - penalty
                for idx in targets:
                    final_scores[idx] = sc
        for indices in zero_groups:
            for idx in indices:
                final_scores[idx] = self.default_reward
        return final_scores

    def compute_scores(self, data, generate_fn):
        prompts, kept_groups, zero_groups, total_size = self.process_input(data)
        outputs = generate_fn(prompts)
        return self.process_output(outputs, kept_groups, zero_groups, total_size)







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


class SeedXRewardModelProcessor:
    def __init__(self, *args, **kwargs):
        self.config = kwargs.get("config")
        self.tokenizer = kwargs.get("tokenizer", None)
        self.input_tokenizer = kwargs.get("input_tokenizer", self.tokenizer)
        self.max_prompt_length = getattr(self.config, "prompt_length", 1 << 20)
        self.extractor_type = self.config.custom_processor.get("extractor_type", "line")
        self.batch_size = getattr(self.config, "seedx_rm_batch_size", 32)
        self.score_scale_factor = getattr(self.config, "score_scale_factor", 1.0)
        self.score_lower_bound = getattr(self.config, "score_lower_bound", -10000.0)
        self.score_upper_bound = getattr(self.config, "score_upper_bound", 10000.0)
        self.overlong_buffer_cfg = self.config.custom_processor.get("overlong_buffer", None)
        self.enable_language_detection = self.config.custom_processor.get("enable_language_detection", False)
        if self.enable_language_detection:
            print(f"Language detection enabled")
        if self.tokenizer is None:
            raise ValueError("tokenizer must be provided")
        if self.input_tokenizer is None:
            raise ValueError("input_tokenizer must be provided")

    def process_input(self, data):
        response_list = _decode_response(data, self.input_tokenizer, self.extractor_type)
        src_text_list = [item["src_text"] for item in data.non_tensor_batch["extra_info"]]
        lang_pair_list = [_get_lang_pair(item) for item in data.non_tensor_batch["extra_info"]]
        src_langs, tgt_langs = zip(*lang_pair_list)
        assert len(src_text_list) == len(response_list) == len(src_langs) == len(tgt_langs)
        prompts: List[str] = []
        chosens: List[str] = []
        kept_indices: List[int] = []
        filtered_indices: List[int] = []
        for idx, (src_text, mt_text, src_lang, tgt_lang) in enumerate(
            zip(src_text_list, response_list, src_langs, tgt_langs)
        ):
            if mt_text is None:
                filtered_indices.append(idx)
                continue
            if self.enable_language_detection:
                if not is_language_match(mt_text, tgt_lang):
                    filtered_indices.append(idx)
                    continue
            prompt = _seedx_build_prompt(src_text, mt_text, src_lang, tgt_lang)
            ids_len = (
                len(self.tokenizer.encode(prompt))
                + len(self.tokenizer.encode(mt_text))
                + 1
            )
            if ids_len > self.max_prompt_length:
                filtered_indices.append(idx)
                continue
            kept_indices.append(idx)
            prompts.append(prompt)
            chosens.append(mt_text)
        total_size = len(src_text_list)
        return prompts, chosens, kept_indices, total_size

    def score_postprocess(self, scores: List[float]) -> List[float]:
        return [
            max(min(score * self.score_scale_factor, self.score_upper_bound), self.score_lower_bound)
            for score in scores
        ]

    def compute_scores(self, data, generate_fn):
        prompts, chosens, kept_indices, total_size = self.process_input(data)
        if len(kept_indices) > 0:
            kept_scores = generate_fn(prompts, chosens)
            default_score = min(kept_scores)
            scores: List[float] = [default_score] * total_size
            for j, idx in enumerate(kept_indices):
                response_ids = data.batch["responses"][idx]
                resp_len = response_ids.shape[-1]
                valid_len = data.batch["attention_mask"][idx][-resp_len:].sum()
                try:
                    valid_len_int = int(valid_len)
                except Exception:
                    valid_len_int = resp_len
                penalty = _compute_overlong_penalty(valid_len_int, self.overlong_buffer_cfg)
                scores[idx] = kept_scores[j] - penalty
        else:
            scores: List[float] = [self.score_lower_bound] * total_size
        return self.score_postprocess(scores)


class VHeadRewardModelProcessor:
    def __init__(self, *args, **kwargs):
        self.config = kwargs.get("config")
        self.tokenizer = kwargs.get("tokenizer", None)
        self.input_tokenizer = kwargs.get("input_tokenizer", self.tokenizer)
        self.max_prompt_length = getattr(self.config, "prompt_length", 1 << 20)
        self.extractor_type = self.config.custom_processor.get("extractor_type", "line")
        self.chat_template = getattr(self.config, "chat_template", True)
        self.batch_size = getattr(self.config, "seedx_rm_batch_size", 32)
        self.score_scale_factor = getattr(self.config, "score_scale_factor", 1.0)
        self.score_lower_bound = getattr(self.config, "score_lower_bound", -10000.0)
        self.score_upper_bound = getattr(self.config, "score_upper_bound", 10000.0)
        self.overlong_buffer_cfg = self.config.custom_processor.get("overlong_buffer", None)
        self.enable_language_detection = self.config.custom_processor.get("enable_language_detection", False)
        if self.enable_language_detection:
            print(f"Language detection enabled")
        if self.tokenizer is None:
            raise ValueError("tokenizer must be provided")
        if self.input_tokenizer is None:
            raise ValueError("input_tokenizer must be provided")

    def process_input(self, data):
        response_list = _decode_response(data, self.input_tokenizer, self.extractor_type)
        src_text_list = [item["src_text"] for item in data.non_tensor_batch["extra_info"]]
        lang_pair_list = [_get_lang_pair(item) for item in data.non_tensor_batch["extra_info"]]
        src_langs, tgt_langs = zip(*lang_pair_list)
        assert len(src_text_list) == len(response_list) == len(src_langs) == len(tgt_langs)
        input_texts = []
        kept_indices = []
        filtered_indices = []
        for idx, (src_text, mt_text, src_lang, tgt_lang) in enumerate(zip(src_text_list, response_list, src_langs, tgt_langs)):
            if mt_text is None:
                filtered_indices.append(idx)
                continue
            if self.enable_language_detection:
                if not is_language_match(mt_text, tgt_lang):
                    filtered_indices.append(idx)
                    continue
            full_text = _vanilla_rm_build_prompt(self.tokenizer, src_lang, tgt_lang, src_text, mt_text, chat_template=self.chat_template)
            ids_len = len(self.tokenizer.encode(full_text))
            if ids_len > self.max_prompt_length:
                filtered_indices.append(idx)
                continue
            kept_indices.append(idx)
            input_texts.append(full_text)
        total_size = len(src_text_list)
        return input_texts, kept_indices, total_size

    def score_postprocess(self, scores: List[float]) -> List[float]:
        return [
            max(min(score * self.score_scale_factor, self.score_upper_bound), self.score_lower_bound)
            for score in scores
        ]

    def compute_scores(self, data, generate_fn):
        input_texts, kept_indices, total_size = self.process_input(data)
        if len(kept_indices) > 0:
            kept_scores = generate_fn(input_texts)
            default_score = min(kept_scores)
            scores: List[float] = [default_score] * total_size
            for j, idx in enumerate(kept_indices):
                response_ids = data.batch["responses"][idx]
                resp_len = response_ids.shape[-1]
                valid_len = data.batch["attention_mask"][idx][-resp_len:].sum()
                try:
                    valid_len_int = int(valid_len)
                except Exception:
                    valid_len_int = resp_len
                penalty = _compute_overlong_penalty(valid_len_int, self.overlong_buffer_cfg)
                scores[idx] = kept_scores[j] - penalty
        else:
            scores: List[float] = [self.score_lower_bound] * total_size
        return self.score_postprocess(scores)




class GroupPostEditRewardProcessor:
    def __init__(self, *args, **kwargs):
        self.config = kwargs.get("config")
        self.tokenizer = kwargs.get("tokenizer", None)
        self.input_tokenizer = kwargs.get("input_tokenizer", self.tokenizer)
        self.max_prompt_length = self.config.prompt_length
        self.extractor_type = self.config.custom_processor.get("extractor_type", "codeblock")
        print(f"Using extractor_type: {self.extractor_type}")
        self.prompt_format = getattr(self.config, "group_prompt_type", "ranking_score")
        self.add_example = getattr(self.config, "group_add_example", False)
        self.reward_scale = getattr(self.config, "reward_scale", 1.0)
        self.default_reward = getattr(self.config, "default_reward", -1.0)
        self.rm_max_candidates = getattr(self.config, "rm_max_candidates", 4)
        self.score_scale_factor = getattr(self.config, "score_scale_factor", 0.1)
        self.overlong_buffer_cfg = self.config.custom_processor.get("overlong_buffer", None)
        self.enable_language_detection = self.config.custom_processor.get("enable_language_detection", False)
        if self.enable_language_detection:
            print(f"Language detection enabled")
        if self.tokenizer is None:
            raise ValueError("tokenizer must be provided")
        if self.input_tokenizer is None:
            raise ValueError("input_tokenizer must be provided")

    def process_input(self, data):
        responses = _decode_response(data, self.input_tokenizer, self.extractor_type)
        extra_info_list = data.non_tensor_batch["extra_info"]
        total_size = len(responses)

        prompt_list: List[Dict[str, List[int]]] = []
        kept_info: List[Dict] = []
        for idx in range(total_size):
            pe_mt_text = responses[idx]
            if pe_mt_text is None:
                continue

            extra = extra_info_list[idx]
            src_lang, tgt_lang = _get_lang_pair(extra)

            if self.enable_language_detection:
                if not is_language_match(pe_mt_text, tgt_lang):
                    continue

            src_text = extra.get("src_text", "")
            mt_texts = extra.get("mt_texts", [])
            notes = extra.get("notes", None)

            if 1 + len(mt_texts) > self.rm_max_candidates:
                mt_texts = mt_texts[: self.rm_max_candidates - 1]

            all_mt_texts = mt_texts + [pe_mt_text]
            num_candidates = len(all_mt_texts)
            if num_candidates < 2:
                continue

            prompt = get_GQM_with_notes_prompt(
                source_lang=src_lang,
                target_lang=tgt_lang,
                source_text=src_text,
                mt_texts=all_mt_texts,
                prompt_format=self.prompt_format,
                add_example=self.add_example,
                notes=notes,
            )
            messages = [{"role": "user", "content": prompt}]
            input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            raw_ids = self.tokenizer.encode(input_text, add_special_tokens=False)

            if len(raw_ids) > self.max_prompt_length:
                continue

            response_ids = data.batch["responses"][idx]
            resp_len = response_ids.shape[-1]
            valid_len = data.batch["attention_mask"][idx][-resp_len:].sum()
            try:
                candidate_len = int(valid_len)
            except Exception:
                candidate_len = resp_len

            kept_info.append({"orig_idx": idx, "num_candidates": num_candidates, "response_len": candidate_len})
            prompt_list.append({"prompt_token_ids": raw_ids})

        return prompt_list, kept_info, total_size

    def process_output(self, outputs, kept_info, total_size) -> List[float]:
        final_scores: List[float] = [self.default_reward] * total_size
        for j, output in enumerate(outputs):
            text = output.outputs[0].text
            info = kept_info[j]
            num_candidates = info["num_candidates"]
            orig_idx = info["orig_idx"]
            scores = group_extract_scores(text, self.prompt_format, num_candidates)
            if scores is None:
                final_scores[orig_idx] = self.default_reward
                continue
            pe_mt_score = scores[-1]
            mean_all = sum(scores) / len(scores)
            relative_reward = (pe_mt_score - mean_all) * self.score_scale_factor
            penalty = _compute_overlong_penalty(info["response_len"], self.overlong_buffer_cfg)
            final_scores[orig_idx] = relative_reward * self.reward_scale - penalty
        return final_scores

    def compute_scores(self, data, generate_fn):
        prompt_list, kept_info, total_size = self.process_input(data)
        outputs = generate_fn(prompt_list)
        return self.process_output(outputs, kept_info, total_size)


def score_reward_fn(data_source, solution_str, ground_truth, extra_info=None):
    return 0
    # print(f"[debug] extra_info: {extra_info}")
    # raise ValueError("extra_info must be provided")



def batch_bleurt_reward_fn(
    data_sources,
    solution_strs,
    ground_truths,
    extra_infos=None,
    extractor_type: str = "line",
    score_scale_factor: float = 1.0,
):
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
    n = len(solution_strs)
    kept_mt: List[str] = []
    kept_ref: List[str] = []
    kept_idx: List[int] = []
    default_score = 0.0
    for i in range(n):
        info = extra_infos[i] if extra_infos is not None else {}
        ref = None
        if isinstance(info, dict):
            ref = info.get("tgt_text") or info.get("trg_text")
        if ref is None:
            ref = ground_truths[i]
        mt_raw = solution_strs[i]
        if isinstance(mt_raw, str):
            if extractor_type == "line":
                mt = _line_extractor(mt_raw)
            elif extractor_type == "codeblock":
                mt = _block_extractor(mt_raw)
            elif extractor_type == "oneline":
                mt = _one_line_extractor(mt_raw)
            else:
                mt = mt_raw
        else:
            mt = None
        if isinstance(mt, str) and isinstance(ref, str) and mt.strip() and ref.strip():
            kept_mt.append(mt)
            kept_ref.append(ref)
            kept_idx.append(i)
    if len(kept_idx) > 0:
        try:
            try:
                from .bleurt_service import func_call as bleurt_func_call
            except Exception:
                from reward_utils.bleurt_service import func_call as bleurt_func_call
            result = bleurt_func_call("BLEURT-20", kept_mt, kept_ref)
            scores_list = result.get("scores", [])
        except Exception:
            scores_list = [default_score] * len(kept_idx)
    else:
        scores_list = []
    final_scores: List[float] = [default_score] * n
    for j, idx in enumerate(kept_idx):
        final_scores[idx] = float(scores_list[j]) * float(score_scale_factor)
    return final_scores


class MultiTaskSelfRewardProcessor:
    """
    Multi-task processor for SelfReward strategy.
    
    Supports two task types:
    1. Translation task: Uses generative reward model (GroupRewardModelProcessor logic)
    2. Ranking task: Uses rule-based scoring (ranking_score_reward_fn)
    
    Data is routed based on `data.non_tensor_batch["ability"]` field:
    - "translation": Uses GroupRewardModelProcessor for generative scoring
    - "ranking": Uses ranking_score_reward_fn for rule-based scoring
    """
    
    def __init__(self, *args, **kwargs):
        self.config = kwargs.get("config")
        self.tokenizer = kwargs.get("tokenizer", None)
        self.input_tokenizer = kwargs.get("input_tokenizer", self.tokenizer)
        
        if self.tokenizer is None:
            raise ValueError("tokenizer must be provided")
        if self.input_tokenizer is None:
            raise ValueError("input_tokenizer must be provided")
        
        self.max_prompt_length = self.config.prompt_length
        self.extractor_type = self.config.custom_processor.get("extractor_type", "line")
        print(f"Using extractor_type: {self.extractor_type}")
        
        self.prompt_type = getattr(self.config, "group_prompt_type", "ranking_score")
        self.add_example = getattr(self.config, "group_add_example", False)
        score_scale_factor  = getattr(self.config, "score_scale_factor", 1.0)
        self.mt_score_scale_factor = getattr(self.config, "mt_score_scale_factor", score_scale_factor)
        self.default_reward = getattr(self.config, "default_reward", 0.0)
        self.overlong_buffer_cfg = self.config.custom_processor.get("overlong_buffer", None)
        self.enable_language_detection = self.config.custom_processor.get("enable_language_detection", False)
        if self.enable_language_detection:
            print(f"Language detection enabled")
        
        self.ranking_score_scale_factor = getattr(self.config, "ranking_score_scale_factor", score_scale_factor)
        
        print(f"MultiTaskSelfRewardProcessor initialized with prompt_type={self.prompt_type}, "
              f"mt_score_scale_factor={self.mt_score_scale_factor}, ranking_score_scale_factor={self.ranking_score_scale_factor}")

    def _split_by_ability(self, data) -> Tuple[List[int], List[int]]:
        """
        Split data indices by ability type.
        
        Returns:
            translation_indices: Indices for translation task
            ranking_indices: Indices for ranking task
        """
        abilities = data.non_tensor_batch.get("ability", None)
        if abilities is None:
            raise ValueError("ability not found in data.non_tensor_batch")
        
        translation_indices = []
        ranking_indices = []
        
        for idx, ability in enumerate(abilities):
            ability_str = str(ability).strip().lower()
            if ability_str == "translation":
                translation_indices.append(idx)
            elif ability_str == "ranking":
                ranking_indices.append(idx)
            else:
                print(f"Warning: Unknown ability type '{ability}' at index {idx}, treating as translation")
                translation_indices.append(idx)
        
        return translation_indices, ranking_indices

    def _group_indices(self, uids: List, indices: List[int]) -> List[Tuple[str, List[int]]]:
        """
        Group indices by uid, only considering indices in the provided list.
        """
        groups: Dict[str, List[int]] = {}
        for idx in indices:
            key = str(uids[idx])
            if key not in groups:
                groups[key] = []
            groups[key].append(idx)
        return list(groups.items())

    def _process_translation_task(self, data, translation_indices: List[int], generate_fn) -> Dict[int, float]:
        """
        Process translation task using GroupRewardModelProcessor logic.
        
        Returns:
            Dict mapping index to score
        """
        if not translation_indices:
            return {}
        
        responses = _decode_response(data, self.input_tokenizer, self.extractor_type)
        extra = data.non_tensor_batch["extra_info"]
        uids = data.non_tensor_batch.get("uid", None)
        if uids is None:
            raise ValueError("uid not found in batch for translation task")
        
        groups = self._group_indices(list(uids), translation_indices)
        
        prompt_list: List[Dict[str, List[int]]] = []
        kept_groups: List[Dict] = []
        zero_groups: List[List[int]] = []

        bad_lang_count = 0
        
        for uid_key, indices in groups:
            src_text = extra[indices[0]]["src_text"]
            src_lang, tgt_lang = _get_lang_pair(extra[indices[0]])
            if len(src_lang) == 2:
                src_lang = LANG_MAP[src_lang]
            if len(tgt_lang) == 2:
                tgt_lang = LANG_MAP[tgt_lang]
            
            seen: Dict[str, int] = {}
            unique_texts: List[str] = []
            dup_map: List[List[int]] = []
            invalid_indices: List[int] = []
            
            for idx in indices:
                t = responses[idx]
                if t is None:
                    invalid_indices.append(idx)
                    continue
                if self.enable_language_detection:
                    if not is_language_match(t, extra[indices[0]]["trg_lang"] if "trg_lang" in extra[indices[0]] else extra[indices[0]]["lang_pair"].split("-")[1]):
                        invalid_indices.append(idx)
                        bad_lang_count += 1
                        continue
                if t in seen:
                    dup_map[seen[t]].append(idx)
                else:
                    seen[t] = len(unique_texts)
                    unique_texts.append(t)
                    dup_map.append([idx])
            
            valid_indices = [i for i in indices if i not in invalid_indices]
            
            if len(unique_texts) <= 1:
                if valid_indices:
                    zero_groups.append(valid_indices)
                for inv in invalid_indices:
                    zero_groups.append([inv])
                continue
            
            prompt = group_get_prompt(src_lang, tgt_lang, src_text, unique_texts, self.prompt_type, add_example=self.add_example)
            messages = [{"role": "user", "content": prompt}]
            input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            raw_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
            
            if len(raw_ids) > self.max_prompt_length:
                if valid_indices:
                    zero_groups.append(valid_indices)
                for inv in invalid_indices:
                    zero_groups.append([inv])
                continue
            
            candidate_lens: List[int] = []
            for targets in dup_map:
                first_idx = targets[0]
                response_ids = data.batch["responses"][first_idx]
                resp_len = response_ids.shape[-1]
                valid_len = data.batch["attention_mask"][first_idx][-resp_len:].sum()
                try:
                    candidate_lens.append(int(valid_len))
                except Exception:
                    candidate_lens.append(resp_len)
            
            kept_groups.append({"uid": [uid_key], "dup_map": dup_map, "candidate_lens": candidate_lens})
            prompt_list.append({"prompt_token_ids": raw_ids})
        
        scores_dict: Dict[int, float] = {}

        if self.enable_language_detection:
            print(f"[DEBUG] Bad lang count: {bad_lang_count}/{len(translation_indices)}")
        
        if prompt_list:
            outputs = generate_fn(prompt_list)
            
            for j, output in enumerate(outputs):
                text = output.outputs[0].text
                group_info = kept_groups[j]
                dup_map = group_info["dup_map"]
                candidate_lens = group_info.get("candidate_lens", [0] * len(dup_map))
                scores = group_extract_scores(text, self.prompt_type, len(dup_map))
                if scores is None:
                    scores = [self.default_reward] * len(dup_map)
                normalized = [s * self.mt_score_scale_factor for s in scores]
                for k, targets in enumerate(dup_map):
                    penalty = _compute_overlong_penalty(candidate_lens[k], self.overlong_buffer_cfg)
                    sc = normalized[k] - penalty
                    for idx in targets:
                        scores_dict[idx] = sc
        
        for indices in zero_groups:
            for idx in indices:
                scores_dict[idx] = self.default_reward
        
        return scores_dict

    def _process_ranking_task(self, data, ranking_indices: List[int]) -> Dict[int, float]:
        """
        Process ranking task using ranking_score_reward_fn.
        
        Returns:
            Dict mapping index to score
        """
        if not ranking_indices:
            return {}
        
        try:
            from reward_utils.ranking_score_reward import ranking_score_reward_fn
        except ImportError:
            from .ranking_score_reward import ranking_score_reward_fn
        
        scores_dict: Dict[int, float] = {}
        
        for idx in ranking_indices:
            response_ids = data.batch["responses"][idx]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch["attention_mask"][idx][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            solution_str = self.input_tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            solution_str = solution_str.replace(self.input_tokenizer.eos_token, "")
            
            ground_truth = None
            if "reward_model" in data.non_tensor_batch:
                reward_model_data = data.non_tensor_batch["reward_model"]
                if isinstance(reward_model_data, dict) and "ground_truth" in reward_model_data:
                    ground_truth = reward_model_data["ground_truth"]
                elif hasattr(reward_model_data, "__getitem__"):
                    try:
                        ground_truth = reward_model_data[idx].get("ground_truth") if isinstance(reward_model_data[idx], dict) else reward_model_data[idx]
                    except (IndexError, KeyError, TypeError):
                        pass
            
            if ground_truth is None:
                print("[Warning] empty ground truth!")
                scores_dict[idx] = self.default_reward
                continue
            
            data_source = data.non_tensor_batch.get("data_source", [""] * len(ranking_indices))
            if isinstance(data_source, (list, tuple)):
                data_source = data_source[idx] if idx < len(data_source) else ""
            
            extra_info = data.non_tensor_batch.get("extra_info", None)
            if extra_info is not None and hasattr(extra_info, "__getitem__"):
                try:
                    extra_info = extra_info[idx]
                except (IndexError, KeyError, TypeError):
                    extra_info = None
            
            reward_result = ranking_score_reward_fn(
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                score_scale_factor=self.ranking_score_scale_factor,
            )
            
            scores_dict[idx] = reward_result.get("score", self.default_reward)
        
        return scores_dict

    def compute_scores(self, data, generate_fn):
        """
        Compute scores for all samples by routing to appropriate task processors.
        
        Args:
            data: DataProto containing batch data
            generate_fn: Function to generate responses (used for translation task)
        
        Returns:
            List of scores for all samples
        """
        total_size = data.batch.batch_size[0]
        translation_indices, ranking_indices = self._split_by_ability(data)
        
        
        final_scores: List[float] = [self.default_reward] * total_size
        
        translation_scores = self._process_translation_task(data, translation_indices, generate_fn)
        for idx, score in translation_scores.items():
            final_scores[idx] = score
        
        ranking_scores = self._process_ranking_task(data, ranking_indices)
        for idx, score in ranking_scores.items():
            final_scores[idx] = score

        avg_translation_score = sum(translation_scores.values())/len(translation_scores)
        avg_ranking_score = sum(ranking_scores.values())/len(ranking_scores)
        print(f"MultiTaskSelfRewardProcessor: {total_size} total, {len(translation_indices)} translation samples, avg score ({avg_translation_score}); "
              f"{len(ranking_indices)} ranking samples, avg score ({avg_ranking_score}).")
        
        return final_scores

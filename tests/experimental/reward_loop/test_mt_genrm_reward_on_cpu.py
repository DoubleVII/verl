import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[3] / "examples" / "rewards" / "mt_genrm_reward.py"
SPEC = importlib.util.spec_from_file_location("mt_genrm_reward", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is True
        assert add_generation_prompt is True
        return list(range(len(messages[0]["content"])))

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))


def test_extract_codeblock_translation_and_score():
    assert MODULE._extract_translation("analysis\n```text\nhello\n```", "codeblock") == "hello"
    assert MODULE._extract_score("analysis\n7") == 7.0
    assert MODULE._extract_score("analysis\n11") is None


def test_language_pair_formats():
    assert MODULE._get_language_pair({"src_lang": "zh", "trg_lang": "en"}) == ("zh", "en")
    assert MODULE._get_language_pair({"lang_pair": "zh-en"}) == ("zh", "en")


@pytest.mark.asyncio
async def test_compute_score_success(monkeypatch):
    async def fake_chat_complete(router_address, request):
        assert router_address == "127.0.0.1:1234"
        assert request["model"] == "/models/genrm"
        return "analysis\n8"

    monkeypatch.setattr(MODULE, "_chat_complete", fake_chat_complete)
    result = await MODULE.compute_score(
        data_source="mt",
        solution_str="analysis\n```\nHello\n```",
        ground_truth=None,
        extra_info={"src_text": "Ni hao", "lang_pair": "zh-en"},
        reward_router_address="127.0.0.1:1234",
        reward_model_tokenizer=FakeTokenizer(),
        model_name="/models/genrm",
        max_prompt_length=1000,
    )
    assert result["score"] == pytest.approx(0.08)
    assert result["genrm_status"] == "ok"
    assert result["genrm_raw_score"] == 8.0


@pytest.mark.asyncio
async def test_failure_results_have_stable_metadata_keys():
    common = {
        "data_source": "mt",
        "ground_truth": None,
        "extra_info": {"src_text": "Ni hao", "lang_pair": "zh-en"},
        "reward_router_address": "127.0.0.1:1234",
        "reward_model_tokenizer": FakeTokenizer(),
        "model_name": "/models/genrm",
    }
    invalid_format = await MODULE.compute_score(solution_str="no code block", **common)
    prompt_too_long = await MODULE.compute_score(
        solution_str="```\nHello\n```", max_prompt_length=1, **common
    )
    assert invalid_format["score"] == -0.04
    assert set(invalid_format) == set(prompt_too_long)

import numpy as np
import torch

from verl import DataProto
from verl.workers.reward_manager.batch import BatchRewardManager

from reward_utils.overlong_reward import compute_overlong_penalty_reward_batch


OVERLONG_CFG = {
    "enable": True,
    "max_resp_len": 100,
    "len": 20,
    "penalty_factor": 1.0,
}


def _score_for_length(length: int) -> dict:
    return compute_overlong_penalty_reward_batch(
        data_sources=["test"],
        solution_strs=[""],
        ground_truths=[None],
        extra_infos=[{"valid_response_length": length}],
        overlong_buffer_cfg=OVERLONG_CFG,
    )[0]


def test_compute_overlong_penalty_reward_batch_below_threshold():
    result = _score_for_length(80)

    assert result["score"] == 0.0
    assert result["overlong_penalty"] == 0.0
    assert result["valid_response_length"] == 80


def test_compute_overlong_penalty_reward_batch_inside_buffer():
    result = _score_for_length(90)

    assert result["score"] == -0.5
    assert result["overlong_penalty"] == 0.5
    assert result["valid_response_length"] == 90


def test_compute_overlong_penalty_reward_batch_at_or_above_max_length():
    result = _score_for_length(100)

    assert result["score"] == -1.0
    assert result["overlong_penalty"] == 1.0
    assert result["valid_response_length"] == 100


class _DummyTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(int(token_id)) for token_id in token_ids)


def test_batch_reward_manager_passes_valid_response_length():
    observed = {}

    def compute_score(data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
        observed["valid_response_length"] = extra_infos[0]["valid_response_length"]
        return [{"score": 0.0}]

    data = DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([[1, 2, 0]]),
            "responses": torch.tensor([[3, 4, 5, 0, 0]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 1, 0, 0]]),
        },
        non_tensors={
            "reward_model": np.array([{"ground_truth": None}], dtype=object),
            "data_source": np.array(["test"], dtype=object),
            "extra_info": np.array([{}], dtype=object),
        },
    )
    reward_manager = BatchRewardManager(
        tokenizer=_DummyTokenizer(),
        num_examine=0,
        compute_score=compute_score,
    )

    reward = reward_manager(data)

    assert observed["valid_response_length"] == 3
    assert reward.shape == data.batch["responses"].shape


def test_batch_reward_manager_does_not_share_mutated_extra_info_between_samples():
    shared_extra_info = {}
    observed_lengths = []

    def compute_score(data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
        observed_lengths.extend(extra_info["valid_response_length"] for extra_info in extra_infos)
        return [{"score": 0.0} for _ in extra_infos]

    data = DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([[1, 2], [1, 2]]),
            "responses": torch.tensor([[3, 4, 5, 0], [6, 7, 8, 9]]),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1],
                ]
            ),
        },
        non_tensors={
            "reward_model": np.array([{"ground_truth": None}, {"ground_truth": None}], dtype=object),
            "data_source": np.array(["test", "test"], dtype=object),
            "extra_info": np.array([shared_extra_info, shared_extra_info], dtype=object),
        },
    )
    reward_manager = BatchRewardManager(
        tokenizer=_DummyTokenizer(),
        num_examine=0,
        compute_score=compute_score,
    )

    reward_manager(data)

    assert observed_lengths == [3, 4]
    assert shared_extra_info == {}

"""Regression coverage for scheduler cache reclamation before GPU output allocation."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.workers.rollout.sglang_rollout import sglang_rollout as module


@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("free_cache_engine", [True, False])
def test_flush_before_output_allocation(monkeypatch, tp_rank, free_cache_engine):
    # Model a separate scheduler allocator: output allocation fails until the
    # scheduler cache is reclaimed. Non-leading TP ranks must not call the engine.
    state = {"cache_live": True}
    raw_output = [{"meta_info": {"output_token_logprobs": [(-0.5, 7, None), (-0.25, 1, None)]}}]

    async def generate(**kwargs):
        return raw_output

    async def flush():
        state["cache_live"] = False
        return True

    engine = SimpleNamespace(async_generate=AsyncMock(side_effect=generate), flush_cache=AsyncMock(side_effect=flush))
    rollout = object.__new__(module.SGLangRollout)
    rollout._tp_rank = tp_rank
    rollout._rank = tp_rank
    rollout._engine = engine if tp_rank == 0 else None
    rollout._device_mesh_cpu = {"tp": SimpleNamespace(get_group=lambda: None, mesh=torch.tensor([0, 1]))}
    rollout.processing_class = SimpleNamespace(pad_token_id=0)
    rollout.pad_token_id = 0
    rollout.sampling_params = {}
    rollout.config = SimpleNamespace(response_length=4, calculate_log_probs=True, free_cache_engine=free_cache_engine)

    def barrier():
        if tp_rank == 0:
            assert not state["cache_live"], "scheduler cache must be reclaimed before the barrier"
        else:
            # The leading TP rank has flushed its scheduler before entering.
            state["cache_live"] = False

    def broadcast(**kwargs):
        if state["cache_live"]:
            raise torch.OutOfMemoryError("scheduler cache exhausted the output allocation budget")
        return [raw_output]

    monkeypatch.setattr(module.dist, "barrier", barrier)
    monkeypatch.setattr(module, "broadcast_pyobj", broadcast)
    monkeypatch.setattr("verl.utils.profiler.performance._get_current_mem_info", lambda: (0, 0, 0, 0))
    prompts = DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor([[0, 2, 3]]),
            "attention_mask": torch.tensor([[0, 1, 1]]),
            "position_ids": torch.tensor([[0, 0, 1]]),
        },
        non_tensors={"raw_prompt_ids": np.array([[2, 3]], dtype=object)},
        meta_info={"eos_token_id": 1},
    )
    loop = asyncio.new_event_loop()
    monkeypatch.setattr(module.asyncio, "get_event_loop", lambda: loop)
    try:
        output = rollout._batch_level_generate_sequences(prompts)
    finally:
        loop.close()

    assert output.batch["input_ids"].tolist() == [[0, 2, 3, 7, 1, 0, 0]]
    assert output.batch["attention_mask"].tolist() == [[0, 1, 1, 1, 1, 0, 0]]
    assert output.batch["rollout_log_probs"].tolist() == [[-0.5, -0.25, 0, 0]]
    assert engine.flush_cache.await_count == (1 if tp_rank == 0 else 0)
    assert engine.async_generate.await_count == (1 if tp_rank == 0 else 0)

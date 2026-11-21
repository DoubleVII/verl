"""CPU regressions for hybrid GenRM inside Ray async actors."""

import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from omegaconf import OmegaConf

from reward_utils.rm_lib import RewardProcessorOutput
from verl import DataProto
from verl.trainer.main_ppo import _select_genrm_reward_model_worker
from verl.workers import fsdp_workers as workers
from verl.workers.fsdp_workers import (
    ActorRolloutRefWorker,
    AsyncGenerativeRewardModelWorker,
    GenerativeRewardModelRolloutWorker,
    GenerativeRewardModelWorker,
)


def run(coro, loop_kind):
    if loop_kind == "uvloop":
        return pytest.importorskip("uvloop").run(coro)
    return asyncio.run(coro)


@pytest.mark.parametrize("loop_kind", ["asyncio", "uvloop"])
def test_async_initialization_builds_sync_engine_without_nested_loop(monkeypatch, loop_kind):
    worker = object.__new__(AsyncGenerativeRewardModelWorker)
    worker.config = OmegaConf.create(
        {
            "model": {},
            "rollout": {
                "name": "vllm",
                "mode": "sync",
                "tensor_model_parallel_size": 1,
                "data_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "load_format": "dummy",
            },
        }
    )
    worker._world_size = 1
    worker._rank = 0
    worker.actor_module_fsdp = object()
    worker._register_dispatch_collect_info = Mock()
    worker.trainer_mode = AsyncMock()
    engine = Mock()
    mesh = Mock()
    mesh.__getitem__ = Mock(return_value=SimpleNamespace(get_local_rank=lambda: 0))
    device = Mock()
    device.get_rng_state.side_effect = ["training_rng", "generation_rng"]
    monkeypatch.setattr(torch.distributed.device_mesh, "init_device_mesh", Mock(return_value=mesh))
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(workers, "fsdp_version", lambda _: 2)
    monkeypatch.setattr(workers, "omega_conf_to_dataclass", lambda cfg, **_: cfg)
    monkeypatch.setattr(workers, "get_torch_device", lambda: device)
    monkeypatch.setattr(workers, "log_gpu_memory_usage", lambda *a, **kw: None)
    rollout_selector = Mock(return_value=Mock(return_value=engine))
    monkeypatch.setattr(workers, "get_rollout_class", rollout_selector)

    def parent_init(self):
        assert asyncio.get_running_loop().is_running()
        self._build_rollout()
        self.trainer_mode.assert_not_awaited()

    monkeypatch.setattr(ActorRolloutRefWorker, "init_model", parent_init)

    async def exercise():
        assert inspect.iscoroutinefunction(worker.init_model)
        await worker.init_model()
        worker.trainer_mode.assert_awaited_once()

    run(exercise(), loop_kind)
    rollout_selector.assert_called_once_with("vllm", "sync")
    assert worker.rollout is engine
    assert worker.torch_random_states == "training_rng"
    assert worker.gen_random_states == "generation_rng"
    assert worker.base_sync_done is False
    assert worker.config.rollout.mode == "sync"


@pytest.fixture
def scoring_worker(monkeypatch):
    worker = object.__new__(AsyncGenerativeRewardModelWorker)
    worker._reward_score_lock = asyncio.Lock()
    worker.generation_config = None
    worker.tokenizer = SimpleNamespace(eos_token_id=1, pad_token_id=0)
    worker.rollout = SimpleNamespace(generate_for_rm=Mock(return_value=[]), release=AsyncMock())
    worker.config = OmegaConf.create({"rollout": {"free_cache_engine": True}})
    worker.actor_module_fsdp = SimpleNamespace(train=Mock())
    worker.torch_random_states = torch.get_rng_state()
    worker.events = []

    async def rollout_mode():
        worker.events.append("rollout")
        await asyncio.sleep(0)

    async def trainer_mode():
        worker.events.append("trainer")
        await ActorRolloutRefWorker.trainer_mode(worker)

    worker.rollout_mode = rollout_mode
    worker.trainer_mode = trainer_mode
    worker.custom_processor = SimpleNamespace(return_reward_model_metadata=False, compute_scores=Mock())
    monkeypatch.setattr(workers, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(
        workers,
        "get_torch_device",
        lambda: SimpleNamespace(
            get_rng_state=torch.get_rng_state,
            set_rng_state=torch.set_rng_state,
            empty_cache=lambda: None,
        ),
    )
    monkeypatch.setattr(workers, "log_gpu_memory_usage", lambda *a, **kw: None)
    monkeypatch.setattr(workers, "aggressive_empty_cache", lambda **kw: None)
    monkeypatch.setattr(workers, "set_expandable_segments", lambda _: None)
    monkeypatch.setattr(workers, "topk_reduce_ratio_min_max", lambda _: (1.0, 1.0, 1.0))
    monkeypatch.setattr(workers, "reduce_timing", lambda timing: timing)
    return worker


def score_data():
    return DataProto.from_dict(
        {
            "responses": torch.zeros((2, 3), dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]),
            "position_ids": torch.arange(5).expand(2, -1),
        },
        meta_info={"collect_reward_model_metadata": True},
    )


@pytest.mark.parametrize("loop_kind", ["asyncio", "uvloop"])
@pytest.mark.parametrize("free_cache", [True, False])
@pytest.mark.parametrize("with_metadata", [True, False])
def test_async_scoring_twice(scoring_worker, loop_kind, free_cache, with_metadata):
    worker = scoring_worker
    worker.config.rollout.free_cache_engine = free_cache

    def compute(data, generate):
        worker.events.append("score")
        assert worker.custom_processor.return_reward_model_metadata
        assert data.meta_info["eos_token_id"] == 1
        generate([])
        if with_metadata:
            return RewardProcessorOutput(scores=[1.0, 2.0], non_tensor_batch={"rm_text": ["a", "b"]})
        return torch.tensor([1.0, 2.0])

    worker.custom_processor.compute_scores.side_effect = compute

    async def exercise():
        assert inspect.iscoroutinefunction(worker.compute_rm_score)
        for _ in range(2):
            output = await worker.compute_rm_score(score_data())
            torch.testing.assert_close(output.batch["rm_scores"], torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]))
            assert output.batch["rm_scores"].device.type == "cpu"
            if with_metadata:
                assert output.non_tensor_batch["rm_text"].tolist() == ["a", "b"]
            assert worker.custom_processor.return_reward_model_metadata is False

    run(exercise(), loop_kind)
    assert worker.events == ["rollout", "score", "trainer"] * 2
    assert worker.rollout.release.await_count == (2 if free_cache else 0)


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_async_scoring_failure_restores_metadata_and_preserves_error(scoring_worker, cleanup_fails):
    worker = scoring_worker
    worker.custom_processor.compute_scores.side_effect = ValueError("processor failed")
    if cleanup_fails:
        worker.rollout.release.side_effect = RuntimeError("cleanup failed")
    with pytest.raises(ValueError, match="processor failed"):
        asyncio.run(worker.compute_rm_score(score_data()))
    assert worker.events == ["rollout", "trainer"]
    assert worker.custom_processor.return_reward_model_metadata is False
    assert not worker._reward_score_lock.locked()


def test_async_cleanup_failure_is_reported(scoring_worker):
    worker = scoring_worker
    worker.custom_processor.compute_scores.return_value = [1.0, 2.0]
    worker.rollout.release.side_effect = RuntimeError("cleanup failed")
    with pytest.raises(RuntimeError, match="cleanup failed"):
        asyncio.run(worker.compute_rm_score(score_data()))


def test_async_rollout_failure_attempts_cleanup(scoring_worker):
    worker = scoring_worker
    worker.rollout_mode = AsyncMock(side_effect=ValueError("wake failed"))
    with pytest.raises(ValueError, match="wake failed"):
        asyncio.run(worker.compute_rm_score(score_data()))
    worker.rollout.release.assert_awaited_once()
    worker.custom_processor.compute_scores.assert_not_called()


def test_async_scoring_serializes_requests(scoring_worker):
    worker = scoring_worker
    worker.custom_processor.compute_scores.side_effect = lambda *a: worker.events.append("score") or [1.0, 2.0]

    async def exercise():
        await asyncio.gather(*(worker.compute_rm_score(score_data()) for _ in range(2)))

    asyncio.run(exercise())
    assert worker.events == ["rollout", "score", "trainer"] * 2


def test_async_worker_selection():
    config = OmegaConf.create({"reward_model": {"strategy": "GenRM"}})
    assert _select_genrm_reward_model_worker(config) is GenerativeRewardModelWorker
    config.reward_model.genrm_worker_mode = "sync"
    assert _select_genrm_reward_model_worker(config) is GenerativeRewardModelWorker
    config.reward_model.genrm_engine_mode = "rollout"
    assert _select_genrm_reward_model_worker(config) is GenerativeRewardModelRolloutWorker
    config.reward_model.genrm_engine_mode = "hybrid"
    config.reward_model.genrm_worker_mode = "async"
    config.reward_model.rollout = {"name": "vllm", "mode": "sync"}
    assert _select_genrm_reward_model_worker(config) is AsyncGenerativeRewardModelWorker


@pytest.mark.parametrize(
    "override",
    [
        {"genrm_worker_mode": "invalid"},
        {"genrm_engine_mode": "rollout"},
        {"rollout": {"name": "sglang", "mode": "sync"}},
        {"rollout": {"name": "vllm", "mode": "async"}},
    ],
)
def test_async_worker_selection_rejects_unsupported_configs(override):
    config = OmegaConf.create(
        {
            "reward_model": {
                "genrm_worker_mode": "async",
                "genrm_engine_mode": "hybrid",
                "rollout": {"name": "vllm", "mode": "sync"},
                **override,
            }
        }
    )
    with pytest.raises(ValueError, match="genrm_worker_mode"):
        _select_genrm_reward_model_worker(config)


def test_async_genrm_in_colocated_ray_actor():
    """Use production binding and async entrypoints with CPU-only model stand-ins."""
    import ray

    from verl.single_controller.base import Worker
    from verl.single_controller.base.decorator import Dispatch, register
    from verl.single_controller.ray.base import RayClassWithInitArgs, create_colocated_worker_cls

    class Policy(Worker):
        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        async def wake_up(self):
            await asyncio.sleep(0)
            return True

    class Reward(AsyncGenerativeRewardModelWorker):
        def __init__(self):
            # All patches live in this disposable Ray actor, never in the driver.
            from unittest.mock import Mock

            from verl.workers import fsdp_workers as remote_workers

            remote_workers.get_device_id = lambda: "cpu"
            remote_workers.get_torch_device = lambda: SimpleNamespace(empty_cache=lambda: None)
            remote_workers.log_gpu_memory_usage = lambda *a, **kw: None
            remote_workers.topk_reduce_ratio_min_max = lambda _: (1.0, 1.0, 1.0)
            remote_workers.reduce_timing = lambda timing: timing
            remote_workers.ActorRolloutRefWorker.init_model = lambda self: self._build_rollout()
            self._reward_score_lock = asyncio.Lock()
            self.events = []
            self.generation_config = None
            self.tokenizer = SimpleNamespace(eos_token_id=1, pad_token_id=0)
            self.rollout = SimpleNamespace(generate_for_rm=lambda prompts: [])
            self.custom_processor = SimpleNamespace(compute_scores=Mock(return_value=[1.0, 2.0]))

        def _build_rollout(self, trust_remote_code=False):
            assert asyncio.get_running_loop().is_running()
            self.events.append("build")

        async def rollout_mode(self):
            await asyncio.sleep(0)
            self.events.append("rollout")

        async def trainer_mode(self):
            await asyncio.sleep(0)
            self.events.append("trainer")

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def get_events(self):
            return self.events

    started_ray = not ray.is_initialized()
    if started_ray:
        ray.init(address="local", num_cpus=2, include_dashboard=False)
    actor = None
    try:
        colocated = create_colocated_worker_cls(
            {
                "policy": RayClassWithInitArgs(cls=ray.remote(Policy)),
                "rm": RayClassWithInitArgs(cls=ray.remote(Reward)),
            }
        )
        actor = colocated.cls.options(
            num_cpus=1,
            num_gpus=0,
            runtime_env={
                "env_vars": {
                    "WORLD_SIZE": "1",
                    "RANK": "0",
                    "MASTER_ADDR": "127.0.0.1",
                    "MASTER_PORT": "29500",
                }
            },
        ).remote()
        ray.get(actor.rm_init_model.remote(), timeout=60)
        assert ray.get(actor.policy_wake_up.remote(), timeout=60)
        for _ in range(2):
            output = ray.get(actor.rm_compute_rm_score.remote(score_data()), timeout=60)
            assert isinstance(output, DataProto)
            torch.testing.assert_close(output.batch["rm_scores"], torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]))
        assert ray.get(actor.rm_get_events.remote(), timeout=60) == [
            "build",
            "trainer",
            "rollout",
            "trainer",
            "rollout",
            "trainer",
        ]
    finally:
        if actor is not None:
            ray.kill(actor)
        if started_ray:
            ray.shutdown()

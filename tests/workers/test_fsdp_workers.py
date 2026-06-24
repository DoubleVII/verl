# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from reward_utils.rm_lib import GENRM_GQM_PROMPTS_KEY, GENRM_GQM_OUTPUTS_KEY, RewardProcessorOutput
from verl.trainer.main_ppo import _select_genrm_reward_model_worker
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.fsdp_workers import (
    ActorRolloutRefWorker,
    GenerativeRewardModelRolloutWorker,
    GenerativeRewardModelWorker,
    _build_genrm_actor_compat_config,
    _build_genrm_rollout_config,
)


def test_actor_rollout_ref_worker_actor_ref_model():
    """Test specifying different reference/actor model"""
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8888"

    config_str = """
    model:
      path: Qwen/Qwen2.5-0.5B-Instruct
    actor:
      _target_: verl.workers.config.FSDPActorConfig
      strategy: fsdp
      fsdp_config:
        _target_: verl.workers.config.FSDPEngineConfig
        fsdp_size: -1
        forward_prefetch: false
      profiler:
        tool: torch_memory
        save_path: ./mem_snapshots
        tool_config:
          torch_memory:
            _target_: verl.utils.profiler.config.TorchMemoryToolConfig
            trace_alloc_max_entries: 100000
            stack_depth: 32
    ref:
      model:
        path: Qwen/Qwen2.5-1.5B-Instruct
      fsdp_config:
        _target_: verl.workers.config.FSDPEngineConfig
        fsdp_size: -1
      profiler:
        tool: torch_memory
        save_path: ./mem_snapshots
        tool_config:
          torch_memory:
            _target_: verl.utils.profiler.config.TorchMemoryToolConfig
            trace_alloc_max_entries: 100000
            stack_depth: 32
      log_prob_micro_batch_size: 1
      ulysses_sequence_parallel_size: 1
      entropy_from_logits_with_chunking: false
    """
    dict_conf = OmegaConf.create(config_str)
    actor_rollout_ref_worker = ActorRolloutRefWorker(dict_conf, role="ref")
    actor_rollout_ref_worker.init_model()

    model_config = actor_rollout_ref_worker.ref_module_fsdp._fsdp_wrapped_module.config
    assert model_config.hidden_size == 1536

    # set ref.model to null, fallback to default case where actor is the same as reference
    dict_conf["ref"]["model"] = None
    actor_rollout_ref_worker = ActorRolloutRefWorker(dict_conf, role="ref")
    actor_rollout_ref_worker.init_model()

    model_config = actor_rollout_ref_worker.ref_module_fsdp._fsdp_wrapped_module.config
    assert model_config.hidden_size == 896


def test_genrm_actor_compat_config_prefers_reward_model_fsdp_config():
    config = OmegaConf.create(
        {
            "strategy": "GenRM",
            "model": {
                "path": "/tmp/reward-model",
                "input_tokenizer": "/tmp/policy-tokenizer",
                "fsdp_config": {"strategy": "fsdp2", "fsdp_size": 4, "forward_prefetch": True},
            },
            "rollout": {"name": "sglang", "tensor_model_parallel_size": 1},
        }
    )
    actor_config = OmegaConf.create(
        {
            "actor": {
                "strategy": "fsdp",
                "ppo_micro_batch_size_per_gpu": 64,
                "profiler": {"tool": "torch_memory"},
                "fsdp_config": {"strategy": "fsdp", "fsdp_size": 8},
            }
        }
    )
    original_config = OmegaConf.to_container(config, resolve=False)
    original_actor_config = OmegaConf.to_container(actor_config, resolve=False)

    merged_cfg = _build_genrm_actor_compat_config(config, actor_config=actor_config)

    assert merged_cfg.actor.strategy == "fsdp2"
    assert merged_cfg.actor.fsdp_config.strategy == "fsdp2"
    assert merged_cfg.actor.fsdp_config.fsdp_size == 4
    assert merged_cfg.actor.fsdp_config.forward_prefetch is True
    assert merged_cfg.actor.ppo_micro_batch_size_per_gpu == 1
    assert merged_cfg.actor.profiler == {}
    assert "model_config" not in merged_cfg.actor
    assert "fsdp_config" not in merged_cfg.model
    assert "input_tokenizer" not in merged_cfg.model

    assert OmegaConf.to_container(config, resolve=False) == original_config
    assert OmegaConf.to_container(actor_config, resolve=False) == original_actor_config


def test_genrm_actor_compat_config_uses_actor_fsdp_as_fallback_only():
    config = OmegaConf.create(
        {
            "strategy": "GenRM",
            "model": {"path": "/tmp/reward-model", "input_tokenizer": None},
            "rollout": {"name": "sglang"},
        }
    )
    actor_config = OmegaConf.create(
        {
            "actor": {
                "strategy": "fsdp",
                "ppo_micro_batch_size_per_gpu": 64,
                "profiler": {"tool": "torch_memory"},
                "fsdp_config": {"strategy": "fsdp", "fsdp_size": 2, "param_offload": True},
            }
        }
    )

    merged_cfg = _build_genrm_actor_compat_config(config, actor_config=actor_config)

    assert merged_cfg.actor.strategy == "fsdp"
    assert merged_cfg.actor.fsdp_config.strategy == "fsdp"
    assert merged_cfg.actor.fsdp_config.fsdp_size == 2
    assert merged_cfg.actor.fsdp_config.param_offload is True
    assert merged_cfg.actor.ppo_micro_batch_size_per_gpu == 1
    assert merged_cfg.actor.profiler == {}


def test_genrm_actor_compat_config_defaults_to_fsdp2_dataclass_config():
    config = OmegaConf.create(
        {
            "strategy": "GenRM",
            "model": {"path": "/tmp/reward-model", "input_tokenizer": None},
            "rollout": {"name": "sglang"},
        }
    )

    merged_cfg = _build_genrm_actor_compat_config(config)
    actor_cfg = omega_conf_to_dataclass(merged_cfg.actor)

    assert merged_cfg.actor.strategy == "fsdp2"
    assert merged_cfg.actor.fsdp_config.strategy == "fsdp2"
    assert merged_cfg.actor.fsdp_config.fsdp_size == -1
    assert actor_cfg.strategy == "fsdp2"
    assert actor_cfg.fsdp_config.strategy == "fsdp2"
    assert actor_cfg.fsdp_config.fsdp_size == -1


def test_genrm_worker_bootstraps_hybrid_rollout_without_loading_model(monkeypatch):
    captured = {}

    def fake_actor_rollout_ref_init(self, config, role, **kwargs):
        captured["config"] = config
        captured["role"] = role
        captured["kwargs"] = kwargs

    def fake_init_tokenizer_processor(self, tokenizer_path, input_tokenizer_path=None):
        captured["tokenizer_path"] = tokenizer_path
        captured["input_tokenizer_path"] = input_tokenizer_path

    monkeypatch.setattr(ActorRolloutRefWorker, "__init__", fake_actor_rollout_ref_init)
    monkeypatch.setattr(GenerativeRewardModelWorker, "init_tokenizer_processor", fake_init_tokenizer_processor)

    config = OmegaConf.create(
        {
            "strategy": "GenRM",
            "model": {
                "path": "/tmp/reward-model",
                "input_tokenizer": "/tmp/policy-tokenizer",
                "fsdp_config": {"strategy": "fsdp2", "fsdp_size": 1},
            },
            "rollout": {"name": "sglang", "tensor_model_parallel_size": 1},
        }
    )

    GenerativeRewardModelWorker(config)

    assert captured["role"] == "actor_rollout"
    assert captured["kwargs"]["disable_optim"] is True
    assert captured["config"].actor.ppo_micro_batch_size_per_gpu == 1
    assert captured["config"].actor.fsdp_config.fsdp_size == 1
    assert "fsdp_config" not in captured["config"].model
    assert "input_tokenizer" not in captured["config"].model
    assert captured["tokenizer_path"] == "/tmp/reward-model"
    assert captured["input_tokenizer_path"] == "/tmp/policy-tokenizer"


def test_genrm_actor_compat_config_carries_distillation_config():
    config = OmegaConf.create(
        {
            "strategy": "GenRM",
            "model": {"path": "/tmp/reward-model", "input_tokenizer": None},
            "rollout": {"name": "sglang"},
        }
    )
    distillation = OmegaConf.create(
        {
            "enabled": True,
            "teacher": {"source": "reward_model", "prompt_source": "reward_model_gqm_out"},
            "distillation_loss": {"loss_mode": "k3"},
        }
    )

    merged_cfg = _build_genrm_actor_compat_config(config, distillation_config=distillation)

    assert merged_cfg.distillation.teacher.source == "reward_model"
    assert merged_cfg.distillation.teacher.prompt_source == "reward_model_gqm_out"


def test_genrm_rollout_config_loads_static_weights_from_model_path():
    config = OmegaConf.create(
        {
            "strategy": "GenRM",
            "model": {
                "path": "/tmp/reward-model",
                "input_tokenizer": "/tmp/policy-tokenizer",
                "fsdp_config": {"strategy": "fsdp2", "fsdp_size": 1},
            },
            "rollout": {"name": "sglang", "load_format": "dummy"},
        }
    )

    merged_cfg = _build_genrm_rollout_config(config)

    assert merged_cfg.rollout.load_format == "auto"
    assert "fsdp_config" not in merged_cfg.model
    assert "input_tokenizer" not in merged_cfg.model
    assert OmegaConf.select(config, "model.input_tokenizer") == "/tmp/policy-tokenizer"
    assert OmegaConf.select(config, "rollout.load_format") == "dummy"


def test_genrm_worker_selection_defaults_to_hybrid_and_supports_rollout():
    config = OmegaConf.create({"reward_model": {"strategy": "GenRM"}})
    assert _select_genrm_reward_model_worker(config) is GenerativeRewardModelWorker

    config.reward_model.genrm_engine_mode = "hybrid"
    assert _select_genrm_reward_model_worker(config) is GenerativeRewardModelWorker

    config.reward_model.genrm_engine_mode = "rollout"
    assert _select_genrm_reward_model_worker(config) is GenerativeRewardModelRolloutWorker


def test_genrm_rollout_worker_bootstraps_without_fsdp_actor(monkeypatch):
    captured = {}

    def fake_worker_init(self):
        self._rank = 0
        self._world_size = 1
        self._local_rank = 0
        self._local_world_size = 1
        self._master_addr = "127.0.0.1"
        self._master_port = "8888"
        self.fused_worker_dict = {}
        self._Worker__dispatch_dp_rank = {}
        self._Worker__collect_dp_rank = {}

    def fake_profiler_init(self, profiler):
        captured["profiler"] = profiler

    def fake_init_tokenizer_processor(self, tokenizer_path, input_tokenizer_path=None):
        captured["tokenizer_path"] = tokenizer_path
        captured["input_tokenizer_path"] = input_tokenizer_path

    monkeypatch.setattr("verl.single_controller.base.Worker.__init__", fake_worker_init)
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr("verl.utils.profiler.DistProfilerExtension.__init__", fake_profiler_init)
    monkeypatch.setattr(
        GenerativeRewardModelRolloutWorker, "init_tokenizer_processor", fake_init_tokenizer_processor
    )

    config = OmegaConf.create(
        {
            "strategy": "GenRM",
            "model": {
                "path": "/tmp/reward-model",
                "input_tokenizer": "/tmp/policy-tokenizer",
                "fsdp_config": {"strategy": "fsdp2", "fsdp_size": 1},
            },
            "rollout": {"name": "sglang", "load_format": "dummy"},
        }
    )

    worker = GenerativeRewardModelRolloutWorker(config, actor_config=OmegaConf.create({"actor": {"unused": True}}))

    assert worker.config.rollout.load_format == "auto"
    assert "fsdp_config" not in worker.config.model
    assert "input_tokenizer" not in worker.config.model
    assert not hasattr(worker, "actor_module_fsdp")
    assert not hasattr(worker, "actor")
    assert captured["tokenizer_path"] == "/tmp/reward-model"
    assert captured["input_tokenizer_path"] == "/tmp/policy-tokenizer"


def test_genrm_rollout_worker_compute_score_uses_rollout_engine_directly(monkeypatch):
    class FakeData:
        def __init__(self):
            self.meta_info = {}

        def to(self, device):
            return self

    class FakeProcessor:
        def __init__(self):
            self.generate_fn = None

        def compute_scores(self, data, generate_fn):
            self.generate_fn = generate_fn
            return torch.tensor([1.0])

    class FakeRollout:
        def __init__(self):
            self.release_calls = 0

        def generate_for_rm(self, prompts):
            return ["ok"]

        async def release(self):
            self.release_calls += 1

    worker = object.__new__(GenerativeRewardModelRolloutWorker)
    worker.config = OmegaConf.create({"rollout": {"free_cache_engine": True}})
    worker.generation_config = None
    worker.tokenizer = SimpleNamespace(eos_token_id=1, pad_token_id=0)
    worker.rollout = FakeRollout()
    worker.custom_processor = FakeProcessor()
    worker._rollout_released = False

    monkeypatch.setattr("verl.workers.fsdp_workers.get_device_id", lambda: 0)
    monkeypatch.setattr("verl.workers.fsdp_workers.get_torch_device", lambda: SimpleNamespace(empty_cache=lambda: None))
    monkeypatch.setattr(
        GenerativeRewardModelRolloutWorker,
        "_expand_to_token_level",
        lambda self, data, scores: torch.zeros((1, 2)),
    )

    output = worker.compute_rm_score(FakeData())

    assert worker.custom_processor.generate_fn.__self__ is worker.rollout
    assert worker.custom_processor.generate_fn.__func__ is worker.rollout.generate_for_rm.__func__
    assert worker.rollout.release_calls == 1
    assert worker._rollout_released is True
    assert "rm_scores" in output.batch


def test_genrm_worker_compute_score_returns_gqm_metadata(monkeypatch):
    class FakeData:
        def __init__(self):
            self.meta_info = {"collect_genrm_gqm_outputs": True}
            self.batch = SimpleNamespace(batch_size=(2,))

        def to(self, device):
            return self

    class FakeProcessor:
        return_gqm_outputs = False

        def compute_scores(self, data, generate_fn):
            assert self.return_gqm_outputs is True
            return RewardProcessorOutput(
                scores=[1.0, 2.0],
                non_tensor_batch={
                    GENRM_GQM_PROMPTS_KEY: [{"prompt_token_ids": [1]}, {"prompt_token_ids": [2]}],
                    GENRM_GQM_OUTPUTS_KEY: ["gqm-a", "gqm-b"],
                },
            )

    worker = object.__new__(GenerativeRewardModelWorker)
    worker._is_actor = False
    worker.generation_config = None
    worker.tokenizer = SimpleNamespace(eos_token_id=1, pad_token_id=0)
    worker.rollout = SimpleNamespace(generate_for_rm=lambda prompts: [])
    worker.custom_processor = FakeProcessor()

    monkeypatch.setattr("verl.workers.fsdp_workers.get_device_id", lambda: 0)
    monkeypatch.setattr("verl.workers.fsdp_workers.get_torch_device", lambda: SimpleNamespace(empty_cache=lambda: None))
    monkeypatch.setattr(
        GenerativeRewardModelWorker,
        "_expand_to_token_level",
        lambda self, data, scores: torch.zeros((2, 3)),
    )
    monkeypatch.setattr("verl.workers.fsdp_workers.topk_reduce_ratio_min_max", lambda _: (1.0, 1.0, 1.0))
    monkeypatch.setattr("verl.workers.fsdp_workers.reduce_timing", lambda timing: timing)

    output = worker.compute_rm_score(FakeData())

    assert "rm_scores" in output.batch
    assert output.non_tensor_batch[GENRM_GQM_PROMPTS_KEY].tolist() == [
        {"prompt_token_ids": [1]},
        {"prompt_token_ids": [2]},
    ]
    assert output.non_tensor_batch[GENRM_GQM_OUTPUTS_KEY].tolist() == ["gqm-a", "gqm-b"]
    assert worker.custom_processor.return_gqm_outputs is False


def test_genrm_worker_distillation_teacher_wraps_actor_logprobs(monkeypatch):
    class FakeData:
        def __init__(self):
            self.meta_info = {}

        def to(self, device):
            return self

    class FakeActor:
        actor_module = object()

        def __init__(self):
            self.seen = None

        def compute_log_prob(self, data, calculate_entropy=False, compute_topk=False):
            self.seen = (data.meta_info.copy(), calculate_entropy, compute_topk)
            return torch.ones((2, 3)), None

    class FakeShardingManager:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

    worker = object.__new__(GenerativeRewardModelWorker)
    worker._is_actor = True
    worker._is_offload_param = False
    worker._world_size = 1
    worker.actor = FakeActor()
    worker.config = OmegaConf.create(
        {
            "actor": {
                "ppo_micro_batch_size_per_gpu": 1,
                "ppo_max_token_len_per_gpu": 16,
                "use_dynamic_bsz": False,
            },
            "rollout": {"temperature": 1.0},
            "distillation": {
                "enabled": True,
                "teacher": {"source": "reward_model", "prompt_source": "reward_model_gqm_out"},
                "distillation_loss": {"loss_mode": "k3"},
            },
        }
    )
    worker.ulysses_sharding_manager = FakeShardingManager()

    output = worker.compute_distillation_teacher_log_prob(FakeData())

    assert torch.equal(output.batch["teacher_logprobs"], torch.ones((2, 3)))
    meta_info, calculate_entropy, compute_topk = worker.actor.seen
    assert calculate_entropy is False
    assert compute_topk is False
    assert meta_info["micro_batch_size"] == 1

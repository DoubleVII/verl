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

from omegaconf import OmegaConf

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.fsdp_workers import (
    ActorRolloutRefWorker,
    GenerativeRewardModelWorker,
    _build_genrm_actor_compat_config,
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

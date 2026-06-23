from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra.errors import InstantiationException
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from verl.trainer.ppo.ray_trainer import Role
from verl.trainer.main_ppo import TaskRunner
from verl.trainer.distillation.losses import compute_sampled_distillation_loss, validate_distillation_config
from verl.utils.config import omega_conf_to_dataclass


def _compose_ppo(overrides=None):
    GlobalHydra.instance().clear()
    try:
        config_dir = str(Path("verl/trainer/config").resolve())
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            return compose(config_name="ppo_trainer", overrides=overrides or [])
    finally:
        GlobalHydra.instance().clear()


def test_distillation_config_disabled_loads():
    cfg = _compose_ppo()
    assert cfg.distillation.enabled is False
    validate_distillation_config(cfg)


def test_ref_policy_teacher_rejects_reference_kl_loss():
    cfg = _compose_ppo(
        [
            "distillation.enabled=True",
            "distillation.teacher.source=ref_policy",
            "actor_rollout_ref.actor.use_kl_loss=True",
        ]
    )
    with pytest.raises(ValueError, match="reference-KL"):
        validate_distillation_config(cfg)


def test_ref_policy_teacher_rejects_reference_kl_reward():
    cfg = _compose_ppo(
        [
            "distillation.enabled=True",
            "distillation.teacher.source=ref_policy",
            "algorithm.use_kl_in_reward=True",
        ]
    )
    with pytest.raises(ValueError, match="reference-KL"):
        validate_distillation_config(cfg)


def test_reserved_teacher_and_topk_loss_error_clearly():
    cfg = _compose_ppo(["distillation.enabled=True", "distillation.teacher.source=reward_model"])
    with pytest.raises(NotImplementedError, match="not implemented yet"):
        validate_distillation_config(cfg)


def test_distillation_teacher_worker_mapping():
    class DummyWorker:
        pass

    current_policy_cfg = _compose_ppo(["distillation.enabled=True", "distillation.teacher.source=current_policy"])
    runner = TaskRunner()
    runner.add_ref_policy_worker(current_policy_cfg, DummyWorker)
    assert Role.RefPolicy not in runner.role_worker_mapping

    ref_policy_cfg = _compose_ppo(["distillation.enabled=True", "distillation.teacher.source=ref_policy"])
    runner = TaskRunner()
    runner.add_ref_policy_worker(ref_policy_cfg, DummyWorker)
    assert Role.RefPolicy in runner.role_worker_mapping
    assert runner.mapping[Role.RefPolicy] == "global_pool"

    cfg = _compose_ppo(["distillation.enabled=True", "distillation.distillation_loss.loss_mode=forward_kl_topk"])
    with pytest.raises(NotImplementedError, match="not supported"):
        validate_distillation_config(cfg)


def test_sampled_k3_loss_is_finite_and_zero_for_identical_logprobs():
    cfg = _compose_ppo(["distillation.enabled=True", "distillation.teacher.source=current_policy"])
    actor_cfg = SimpleNamespace(loss_agg_mode="token-mean")
    distillation_cfg = omega_conf_to_dataclass(cfg.distillation)
    log_prob = torch.tensor([[-1.0, -2.0], [-0.5, -1.5]])
    response_mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.bool)

    loss, metrics = compute_sampled_distillation_loss(
        config=actor_cfg,
        distillation_config=distillation_cfg,
        log_prob=log_prob,
        teacher_logprobs=log_prob.clone(),
        response_mask=response_mask,
        old_log_prob=log_prob.detach(),
    )

    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert metrics["distillation/abs_loss"] == pytest.approx(0.0, abs=1e-6)


def test_direct_backprop_k1_is_rejected():
    cfg = _compose_ppo(
        [
            "distillation.enabled=True",
            "distillation.teacher.source=current_policy",
            "distillation.distillation_loss.loss_mode=k1",
            "distillation.distillation_loss.use_policy_gradient=False",
        ]
    )
    with pytest.raises(InstantiationException, match="Directly backpropagating"):
        omega_conf_to_dataclass(cfg.distillation)

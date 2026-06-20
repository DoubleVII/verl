from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra.errors import InstantiationException
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from verl.trainer.ppo.ray_trainer import Role
from verl.trainer.main_ppo import TaskRunner
from verl.trainer.distillation.losses import (
    compute_forward_kl_topk_distillation_loss,
    compute_forward_kl_topk_distillation_loss_flat,
    compute_sampled_distillation_loss,
    compute_topk_kl_losses,
    compute_topk_logprobs_from_logits,
    validate_distillation_config,
)
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


def test_reserved_teacher_error_clearly():
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


def test_forward_kl_topk_config_loads_for_fsdp():
    cfg = _compose_ppo(
        [
            "distillation.enabled=True",
            "distillation.teacher.source=current_policy",
            "distillation.distillation_loss.loss_mode=forward_kl_topk",
            "distillation.distillation_loss.topk=4",
        ]
    )
    validate_distillation_config(cfg)
    distillation_cfg = omega_conf_to_dataclass(cfg.distillation)
    assert distillation_cfg.distillation_loss.loss_mode == "forward_kl_topk"


def test_forward_kl_topk_reverse_config_loads_for_fsdp():
    cfg = _compose_ppo(
        [
            "distillation.enabled=True",
            "distillation.teacher.source=current_policy",
            "distillation.distillation_loss.loss_mode=forward_kl_topk",
            "distillation.distillation_loss.topk=4",
            "distillation.distillation_loss.topk_kl_mode=reverse",
        ]
    )
    validate_distillation_config(cfg)
    distillation_cfg = omega_conf_to_dataclass(cfg.distillation)
    assert distillation_cfg.distillation_loss.topk_kl_mode == "reverse"


def test_forward_kl_topk_rejects_invalid_topk_kl_mode():
    cfg = _compose_ppo(
        [
            "distillation.enabled=True",
            "distillation.distillation_loss.loss_mode=forward_kl_topk",
            "distillation.distillation_loss.topk_kl_mode=symmetric",
        ]
    )
    with pytest.raises(InstantiationException, match="topk_kl_mode"):
        omega_conf_to_dataclass(cfg.distillation)


def test_forward_kl_topk_rejects_missing_topk():
    cfg = _compose_ppo(
        [
            "distillation.enabled=True",
            "distillation.distillation_loss.loss_mode=forward_kl_topk",
            "distillation.distillation_loss.topk=null",
        ]
    )
    with pytest.raises(InstantiationException, match="topk must be set"):
        omega_conf_to_dataclass(cfg.distillation)


def test_forward_kl_topk_rejects_megatron():
    cfg = _compose_ppo(
        [
            "distillation.enabled=True",
            "distillation.distillation_loss.loss_mode=forward_kl_topk",
            "actor_rollout_ref.actor.strategy=megatron",
        ]
    )
    with pytest.raises(NotImplementedError, match="FSDP only"):
        validate_distillation_config(cfg)


def test_forward_kl_topk_loss_is_finite_and_zero_for_identical_topk():
    cfg = _compose_ppo(
        [
            "distillation.enabled=True",
            "distillation.teacher.source=current_policy",
            "distillation.distillation_loss.loss_mode=forward_kl_topk",
            "distillation.distillation_loss.topk=2",
        ]
    )
    actor_cfg = SimpleNamespace(loss_agg_mode="token-mean")
    distillation_cfg = omega_conf_to_dataclass(cfg.distillation)
    logits = torch.tensor(
        [
            [[3.0, 1.0, -1.0], [0.5, 2.0, -0.5]],
            [[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]],
        ]
    )
    teacher_logprobs, teacher_ids = compute_topk_logprobs_from_logits(logits=logits, topk=2)
    response_mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.bool)

    loss, metrics = compute_forward_kl_topk_distillation_loss(
        config=actor_cfg,
        distillation_config=distillation_cfg,
        student_logits=logits,
        teacher_logprobs=teacher_logprobs,
        teacher_ids=teacher_ids,
        response_mask=response_mask,
    )

    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert metrics["distillation/student_mass"] == pytest.approx(metrics["distillation/teacher_mass"], abs=1e-6)


def test_forward_kl_topk_reverse_loss_uses_student_weights():
    teacher_logprobs = torch.log(torch.tensor([[0.7, 0.2], [0.4, 0.3]]))
    student_logprobs = torch.log(torch.tensor([[0.5, 0.4], [0.6, 0.1]]))

    reverse_losses = compute_topk_kl_losses(
        teacher_logprobs=teacher_logprobs,
        student_logprobs=student_logprobs,
        topk_kl_mode="reverse",
    )
    expected_reverse = (student_logprobs.exp() * (student_logprobs - teacher_logprobs)).sum(dim=-1)
    forward_losses = compute_topk_kl_losses(
        teacher_logprobs=teacher_logprobs,
        student_logprobs=student_logprobs,
        topk_kl_mode="forward",
    )

    assert torch.allclose(reverse_losses, expected_reverse)
    assert not torch.allclose(reverse_losses, forward_losses)


def test_forward_kl_topk_flat_loss_matches_padded_loss():
    cfg = _compose_ppo(
        [
            "distillation.enabled=True",
            "distillation.teacher.source=current_policy",
            "distillation.distillation_loss.loss_mode=forward_kl_topk",
            "distillation.distillation_loss.topk=2",
        ]
    )
    actor_cfg = SimpleNamespace(loss_agg_mode="token-mean")
    distillation_cfg = omega_conf_to_dataclass(cfg.distillation)
    student_logits = torch.tensor(
        [
            [[3.0, 1.0, -1.0], [0.5, 2.0, -0.5]],
            [[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]],
        ]
    )
    teacher_logits = student_logits + torch.tensor([[[0.2, -0.1, 0.0], [0.0, 0.1, -0.2]], [[0.1, 0.0, -0.1], [0.0, 0.0, 0.0]]])
    teacher_logprobs, teacher_ids = compute_topk_logprobs_from_logits(logits=teacher_logits, topk=2)
    response_mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.bool)

    padded_loss, _ = compute_forward_kl_topk_distillation_loss(
        config=actor_cfg,
        distillation_config=distillation_cfg,
        student_logits=student_logits,
        teacher_logprobs=teacher_logprobs,
        teacher_ids=teacher_ids,
        response_mask=response_mask,
    )
    flat_loss, _ = compute_forward_kl_topk_distillation_loss_flat(
        config=actor_cfg,
        distillation_config=distillation_cfg,
        student_logits=student_logits[response_mask],
        teacher_logprobs=teacher_logprobs,
        teacher_ids=teacher_ids,
        response_mask=response_mask,
    )

    assert flat_loss.item() == pytest.approx(padded_loss.item(), abs=1e-6)


def test_forward_kl_topk_flat_loss_uses_response_mask_alignment():
    cfg = _compose_ppo(
        [
            "distillation.enabled=True",
            "distillation.teacher.source=current_policy",
            "distillation.distillation_loss.loss_mode=forward_kl_topk",
            "distillation.distillation_loss.topk=2",
        ]
    )
    actor_cfg = SimpleNamespace(loss_agg_mode="token-mean")
    distillation_cfg = omega_conf_to_dataclass(cfg.distillation)
    student_logits = torch.tensor(
        [
            [[3.0, 1.0, -1.0], [0.5, 2.0, -0.5], [1.0, 0.0, 2.0]],
            [[0.0, 1.0, 3.0], [2.0, 0.0, 1.0], [0.0, 3.0, 1.0]],
        ]
    )
    teacher_logits = student_logits + 0.1
    teacher_logprobs, teacher_ids = compute_topk_logprobs_from_logits(logits=teacher_logits, topk=2)
    response_mask = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool)

    padded_loss, _ = compute_forward_kl_topk_distillation_loss(
        config=actor_cfg,
        distillation_config=distillation_cfg,
        student_logits=student_logits,
        teacher_logprobs=teacher_logprobs,
        teacher_ids=teacher_ids,
        response_mask=response_mask,
    )
    flat_loss, _ = compute_forward_kl_topk_distillation_loss_flat(
        config=actor_cfg,
        distillation_config=distillation_cfg,
        student_logits=student_logits[response_mask],
        teacher_logprobs=teacher_logprobs,
        teacher_ids=teacher_ids,
        response_mask=response_mask,
    )

    assert flat_loss.item() == pytest.approx(padded_loss.item(), abs=1e-6)


def test_reverse_kl_topk_flat_loss_matches_padded_loss():
    cfg = _compose_ppo(
        [
            "distillation.enabled=True",
            "distillation.teacher.source=current_policy",
            "distillation.distillation_loss.loss_mode=forward_kl_topk",
            "distillation.distillation_loss.topk=2",
            "distillation.distillation_loss.topk_kl_mode=reverse",
        ]
    )
    actor_cfg = SimpleNamespace(loss_agg_mode="token-mean")
    distillation_cfg = omega_conf_to_dataclass(cfg.distillation)
    student_logits = torch.tensor(
        [
            [[3.0, 1.0, -1.0], [0.5, 2.0, -0.5]],
            [[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]],
        ]
    )
    teacher_logits = student_logits + torch.tensor(
        [[[0.2, -0.1, 0.0], [0.0, 0.1, -0.2]], [[0.1, 0.0, -0.1], [0.0, 0.0, 0.0]]]
    )
    teacher_logprobs, teacher_ids = compute_topk_logprobs_from_logits(logits=teacher_logits, topk=2)
    response_mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.bool)

    padded_loss, _ = compute_forward_kl_topk_distillation_loss(
        config=actor_cfg,
        distillation_config=distillation_cfg,
        student_logits=student_logits,
        teacher_logprobs=teacher_logprobs,
        teacher_ids=teacher_ids,
        response_mask=response_mask,
    )
    flat_loss, _ = compute_forward_kl_topk_distillation_loss_flat(
        config=actor_cfg,
        distillation_config=distillation_cfg,
        student_logits=student_logits[response_mask],
        teacher_logprobs=teacher_logprobs,
        teacher_ids=teacher_ids,
        response_mask=response_mask,
    )

    assert flat_loss.item() == pytest.approx(padded_loss.item(), abs=1e-6)

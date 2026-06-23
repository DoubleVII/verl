from types import SimpleNamespace

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _trainer(shared_first_turn_by_uid: bool):
    trainer = object.__new__(RayPPOTrainer)
    trainer.async_rollout_mode = False
    trainer.use_distillation = False
    trainer.distillation_data_teacher_prompt = False
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(
                multi_turn=SimpleNamespace(shared_first_turn_by_uid=shared_first_turn_by_uid)
            )
        )
    )
    return trainer


def _batch():
    tensors = TensorDict(
        {
            "input_ids": torch.ones((2, 4), dtype=torch.long),
            "attention_mask": torch.ones((2, 4), dtype=torch.long),
            "position_ids": torch.arange(4, dtype=torch.long).repeat(2, 1),
        },
        batch_size=2,
    )
    return DataProto(
        batch=tensors,
        non_tensor_batch={
            "raw_prompt": np.array([[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]], dtype=object),
            "interaction_kwargs": np.array([{"name": "gqm_post_edit"}, {"name": "gqm_post_edit"}], dtype=object),
            "uid": np.array(["u0", "u1"], dtype=object),
            "ability": np.array(["gqm_post_edit", "gqm_post_edit"], dtype=object),
        },
    )


def test_get_gen_batch_copies_uid_for_shared_first_turn_without_removing_from_train_batch():
    batch = _batch()
    gen_batch = _trainer(shared_first_turn_by_uid=True)._get_gen_batch(batch)

    assert gen_batch.non_tensor_batch["uid"].tolist() == ["u0", "u1"]
    assert batch.non_tensor_batch["uid"].tolist() == ["u0", "u1"]
    assert "ability" in batch.non_tensor_batch
    assert "ability" not in gen_batch.non_tensor_batch


def test_get_gen_batch_keeps_existing_behavior_when_shared_first_turn_disabled():
    batch = _batch()
    gen_batch = _trainer(shared_first_turn_by_uid=False)._get_gen_batch(batch)

    assert "uid" not in gen_batch.non_tensor_batch
    assert batch.non_tensor_batch["uid"].tolist() == ["u0", "u1"]

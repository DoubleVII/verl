from verl.workers.rollout.sglang_rollout.sglang_rollout import _normalize_interaction_kwargs


def test_normalize_interaction_kwargs_none_to_empty_dict():
    assert _normalize_interaction_kwargs(None) == {}


def test_normalize_interaction_kwargs_null_name_to_empty_dict():
    assert _normalize_interaction_kwargs({"name": None}) == {}


def test_normalize_interaction_kwargs_preserves_valid_name():
    assert _normalize_interaction_kwargs({"name": "gqm_post_edit"}) == {"name": "gqm_post_edit"}

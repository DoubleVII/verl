from verl.utils.dataset.rl_dataset import get_interaction_kwargs


def test_gqm_post_edit_gets_default_interaction_kwargs():
    row = {"ability": "gqm_post_edit", "extra_info": {}}

    assert get_interaction_kwargs(row) == {"name": "gqm_post_edit"}


def test_existing_interaction_kwargs_are_preserved():
    row = {
        "ability": "gqm_post_edit",
        "extra_info": {"interaction_kwargs": {"name": "custom", "prompt": "use this"}},
    }

    assert get_interaction_kwargs(row) == {"name": "custom", "prompt": "use this"}


def test_other_abilities_do_not_get_interaction_kwargs():
    row = {"ability": "ranking", "extra_info": {}}

    assert get_interaction_kwargs(row) == {}

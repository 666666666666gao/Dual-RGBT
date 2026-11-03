import numpy as np
import pytest

from dual_rgbt.c2_data import sample_full_timeline_pair
from dual_rgbt.data import sample_lasher_pair


def test_samples_later_timeline_without_future_template():
    valid = [0, 100, 200, 300, 400, 500, 600]
    selected = set()
    for seed in range(100):
        first, recent, search = sample_full_timeline_pair(valid, np.random.RandomState(seed))
        assert first == 0
        assert recent in valid and recent < search and search - recent <= 200
        selected.add(search)
    assert selected == set(valid[1:])


def test_original_prefix_recipe_remains_separate():
    valid = [0, 100, 200, 300, 400]
    old = {sample_lasher_pair(valid, np.random.RandomState(seed))[2]
           for seed in range(100)}
    new = {sample_full_timeline_pair(valid, np.random.RandomState(seed))[2]
           for seed in range(100)}
    assert old == {100, 200}
    assert new == {100, 200, 300, 400}


def test_gapped_search_requires_a_recent_valid_predecessor():
    valid = [3, 5, 204, 205, 500, 501]
    selected = set()
    for seed in range(100):
        first, recent, search = sample_full_timeline_pair(valid, np.random.RandomState(seed))
        assert first == 3 and 0 < search - recent <= 200
        selected.add(search)
    assert selected == {5, 204, 205, 501}


def test_interval_boundary_is_inclusive():
    assert sample_full_timeline_pair([8, 208], np.random.RandomState(0)) == (8, 8, 208)


def test_rejects_no_causal_pair():
    with pytest.raises(AssertionError):
        sample_full_timeline_pair([8, 209], np.random.RandomState(0))


def test_seed_reproduces_pair():
    valid = list(range(0, 1600, 20))
    assert sample_full_timeline_pair(valid, np.random.RandomState(2027)) == (
        sample_full_timeline_pair(valid, np.random.RandomState(2027)))

import itertools

import numpy as np
import pytest

from src.attacks.moeva2.utils import get_one_hot_encoding_constraints


class TestUtils:
    @pytest.mark.parametrize(
        "type_mask,x,oracle",
        [
            (["int"], np.array([[0.0], [0.0]]), np.array([0.0, 0.0])),
            (["ohe0"], np.array([[0.0], [1.0]]), np.array([1.0, 0.0])),
            (
                ["ohe0", "ohe1"],
                np.array(list(itertools.product([0, 1], repeat=2))),
                np.array([2.0, 1.0, 1.0, 0.0]),
            ),
            (
                ["ohe0", "ohe0", "ohe1", "ohe1", "ohe1"],
                np.array(list(itertools.product([0, 1], repeat=5))),
                np.array(
                    [
                        2,
                        1,
                        1,
                        2,
                        1,
                        2,
                        2,
                        3,
                        1,
                        0,
                        0,
                        1,
                        0,
                        1,
                        1,
                        2,
                        1,
                        0,
                        0,
                        1,
                        0,
                        1,
                        1,
                        2,
                        2,
                        1,
                        1,
                        2,
                        1,
                        2,
                        2,
                        3,
                    ]
                ),
            ),
            (
                ["int", "ohe0", "ohe0", "ohe1", "ohe1", "ohe1"],
                np.concatenate(
                    (
                        np.arange(32).reshape(-1, 1),
                        np.array(list(itertools.product([0, 1], repeat=5))),
                    ),
                    axis=1,
                ),
                np.array(
                    [
                        2,
                        1,
                        1,
                        2,
                        1,
                        2,
                        2,
                        3,
                        1,
                        0,
                        0,
                        1,
                        0,
                        1,
                        1,
                        2,
                        1,
                        0,
                        0,
                        1,
                        0,
                        1,
                        1,
                        2,
                        2,
                        1,
                        1,
                        2,
                        1,
                        2,
                        2,
                        3,
                    ]
                ),
            ),
        ],
    )
    def test_get_one_hot_encoding_constraints(self, type_mask, x, oracle):
        assert np.array_equal(get_one_hot_encoding_constraints(type_mask, x), oracle)

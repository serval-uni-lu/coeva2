import pytest
import numpy as np

from src.attacks.moeva2.feature_encoder import FeatureEncoder


@pytest.fixture(scope="function")
def feature_encoder():
    mutable_mask = ["True", "False", "True"]
    type_mask = ["int", "real", "real"]
    xl = np.array([0.0, 0.0])

    return True


class TestFeatureEncoder:
    def test_smoke(self, feature_encoder):
        pass
        # assert "a" in feature_encoder

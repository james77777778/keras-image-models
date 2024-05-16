import pytest
from absl.testing import parameterized
from keras.src import testing

from kimm._src.layers.learnable_affine import LearnableAffine


class LearnableAffineTest(testing.TestCase, parameterized.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_layer_scale_basic(self):
        self.run_layer_test(
            LearnableAffine,
            init_kwargs={"scale_value": 1.0, "bias_value": 0.0},
            input_shape=(1, 10),
            expected_output_shape=(1, 10),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_losses=0,
            supports_masking=False,
        )

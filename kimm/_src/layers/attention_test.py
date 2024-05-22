import keras
import pytest
from absl.testing import parameterized
from keras.src import testing

from kimm._src.layers.attention import Attention


class AttentionTest(testing.TestCase, parameterized.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_basic_3d(self):
        self.run_layer_test(
            Attention,
            init_kwargs={"hidden_dim": 20, "num_heads": 2},
            input_shape=(1, 10, 20),
            expected_output_shape=(1, 10, 20),
            expected_num_trainable_weights=3,
            expected_num_non_trainable_weights=0,
            expected_num_losses=0,
            supports_masking=False,
        )

    @pytest.mark.requires_trainable_backend
    def test_basic_4d(self):
        self.run_layer_test(
            Attention,
            init_kwargs={"hidden_dim": 20, "num_heads": 2},
            input_shape=(1, 2, 10, 20),
            expected_output_shape=(1, 2, 10, 20),
            expected_num_trainable_weights=3,
            expected_num_non_trainable_weights=0,
            expected_num_losses=0,
            supports_masking=False,
        )

    def test_invalid_ndim(self):
        # Test 2D
        inputs = keras.Input(shape=[1])
        with self.assertRaisesRegex(
            ValueError, "The ndim of the inputs must be 3 or 4."
        ):
            Attention(1, 1)(inputs)

        # Test 5D
        inputs = keras.Input(shape=[1, 2, 3, 4])
        with self.assertRaisesRegex(
            ValueError, "The ndim of the inputs must be 3 or 4."
        ):
            Attention(1, 1)(inputs)

import pytest
from absl.testing import parameterized
from keras.src import testing

from kimm.layers.attention import Attention


class AttentionTest(testing.TestCase, parameterized.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_attention_basic(self):
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

import pytest
from absl.testing import parameterized
from keras import layers
from keras.src import testing

from kimm.layers.position_embedding import PositionEmbedding


class PositionEmbeddingTest(testing.TestCase, parameterized.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_position_embedding_basic(self):
        self.run_layer_test(
            PositionEmbedding,
            init_kwargs={},
            input_shape=(1, 10, 10),
            expected_output_shape=(1, 11, 10),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_losses=0,
            supports_masking=False,
        )

    @pytest.mark.requires_trainable_backend
    def test_position_embedding_invalid_input_shape(self):
        inputs = layers.Input([3])
        with self.assertRaisesRegex(
            ValueError, "PositionEmbedding only accepts 3-dimensional input."
        ):
            PositionEmbedding()(inputs)

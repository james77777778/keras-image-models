import pytest
from absl.testing import parameterized
from keras import layers
from keras import models
from keras.src import testing

from kimm._src.layers.position_embedding import PositionEmbedding


class PositionEmbeddingTest(testing.TestCase, parameterized.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_basic(self):
        self.run_layer_test(
            PositionEmbedding,
            init_kwargs={"height": 2, "width": 5},
            input_shape=(1, 10, 10),
            expected_output_shape=(1, 11, 10),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_losses=0,
            supports_masking=False,
        )

    def test_embedding_resizing(self):
        temp_dir = self.get_temp_dir()
        model = models.Sequential(
            [layers.Input(shape=[256, 8]), PositionEmbedding(16, 16)]
        )
        model.save(f"{temp_dir}/model.keras")

        # Resize from (16, 16) to (8, 8)
        model = models.Sequential(
            [layers.Input(shape=[64, 8]), PositionEmbedding(8, 8)]
        )
        model.load_weights(f"{temp_dir}/model.keras")

    @pytest.mark.requires_trainable_backend
    def test_invalid_input_shape(self):
        inputs = layers.Input([3])
        with self.assertRaisesRegex(
            ValueError, "PositionEmbedding only accepts 3-dimensional input."
        ):
            PositionEmbedding(2, 2)(inputs)

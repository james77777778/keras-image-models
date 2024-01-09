from absl.testing import parameterized
from keras import random
from keras.src import testing

from kimm.layers.position_embedding import PositionEmbedding


class PositionEmbeddingTest(testing.TestCase, parameterized.TestCase):
    def test_position_embedding(self):
        x = random.uniform([1, 123, 768])
        layer = PositionEmbedding()

        y = layer(x)

        self.assertEqual(y.shape, [1, 124, 768])

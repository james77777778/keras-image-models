from absl.testing import parameterized
from keras import random
from keras.src import testing

from kimm.layers.attention import Attention


class AttentionTest(testing.TestCase, parameterized.TestCase):
    def test_attention(self):
        x = random.uniform([1, 197, 768])
        layer = Attention(768)

        y = layer(x)

        self.assertEqual(y.shape, [1, 197, 768])

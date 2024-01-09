from absl.testing import parameterized
from keras import random
from keras.src import testing

from kimm.layers.layer_scale import LayerScale


class LayerScaleTest(testing.TestCase, parameterized.TestCase):
    def test_layer_scale(self):
        x = random.uniform([1, 123])
        layer = LayerScale(123)

        y = layer(x)

        self.assertEqual(y.shape, [1, 123])

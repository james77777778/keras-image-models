import pytest
from absl.testing import parameterized
from keras import models
from keras import random
from keras.src import testing

from kimm.models.convmixer import ConvMixer736D32


class ConvMixerTest(testing.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        [(ConvMixer736D32.__name__, ConvMixer736D32)]
    )
    def test_convmixer_base(self, model_class):
        # TODO: test the correctness of the real image
        x = random.uniform([1, 224, 224, 3]) * 255.0
        model = model_class(input_shape=[224, 224, 3])

        y = model(x, training=False)

        self.assertEqual(y.shape, (1, 1000))

    @parameterized.named_parameters(
        [(ConvMixer736D32.__name__, ConvMixer736D32, 32, 768)]
    )
    def test_convmixer_feature_extractor(
        self, model_class, patch_size, hidden_channels
    ):
        x = random.uniform([1, 224, 224, 3]) * 255.0
        model = model_class(input_shape=[224, 224, 3], feature_extractor=True)

        y = model(x, training=False)

        self.assertIsInstance(y, dict)
        self.assertContainsSubset(
            model_class.available_feature_keys(),
            list(y.keys()),
        )
        self.assertEqual(
            list(y["STEM"].shape), [1, patch_size, patch_size, hidden_channels]
        )
        self.assertEqual(
            list(y["BLOCK0"].shape),
            [1, patch_size, patch_size, hidden_channels],
        )
        self.assertEqual(
            list(y["BLOCK1"].shape),
            [1, patch_size, patch_size, hidden_channels],
        )
        self.assertEqual(
            list(y["BLOCK2"].shape),
            [1, patch_size, patch_size, hidden_channels],
        )
        self.assertEqual(
            list(y["BLOCK3"].shape),
            [1, patch_size, patch_size, hidden_channels],
        )

    @pytest.mark.serialization
    @parameterized.named_parameters(
        [(ConvMixer736D32.__name__, ConvMixer736D32, 224)]
    )
    def test_convmixer_serialization(self, model_class, image_size):
        x = random.uniform([1, image_size, image_size, 3]) * 255.0
        temp_dir = self.get_temp_dir()
        model1 = model_class(input_shape=[224, 224, 3])
        y1 = model1(x, training=False)
        model1.save(temp_dir + "/model.keras")

        model2 = models.load_model(temp_dir + "/model.keras")
        y2 = model2(x, training=False)

        self.assertAllClose(y1, y2)

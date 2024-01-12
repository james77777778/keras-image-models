from absl.testing import parameterized
from keras import random
from keras.src import testing

from kimm.models.efficientnet import EfficientNetB0
from kimm.models.efficientnet import EfficientNetB2
from kimm.utils import make_divisible


class EfficientNetTest(testing.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        [
            (EfficientNetB0.__name__, EfficientNetB0, 224),
            (EfficientNetB2.__name__, EfficientNetB2, 260),
        ]
    )
    def test_efficentnet_base(self, model_class, image_size):
        # TODO: test the correctness of the real image
        x = random.uniform([1, image_size, image_size, 3]) * 255.0
        model = model_class()

        y = model.predict(x)

        self.assertEqual(y.shape, (1, 1000))

    @parameterized.named_parameters(
        [
            (EfficientNetB0.__name__, EfficientNetB0, 1.0),
            (EfficientNetB2.__name__, EfficientNetB2, 1.1),
        ]
    )
    def test_efficentnet_feature_extractor(self, model_class, width):
        x = random.uniform([1, 224, 224, 3]) * 255.0
        model = model_class(
            input_shape=[224, 224, 3], as_feature_extractor=True
        )

        y = model.predict(x)

        self.assertIsInstance(y, dict)
        self.assertAllEqual(
            list(y.keys()), model_class.available_feature_keys()
        )
        self.assertEqual(
            list(y["STEM_S2"].shape), [1, 112, 112, make_divisible(32 * width)]
        )
        self.assertEqual(
            list(y["BLOCK1_S4"].shape), [1, 56, 56, make_divisible(24 * width)]
        )
        self.assertEqual(
            list(y["BLOCK2_S8"].shape), [1, 28, 28, make_divisible(40 * width)]
        )
        self.assertEqual(
            list(y["BLOCK3_S16"].shape), [1, 14, 14, make_divisible(80 * width)]
        )
        self.assertEqual(
            list(y["BLOCK5_S32"].shape), [1, 7, 7, make_divisible(192 * width)]
        )

from absl.testing import parameterized
from keras import random
from keras.src import testing

from kimm.models.mobilenet_v3 import MobileNet100V3Large
from kimm.models.mobilenet_v3 import MobileNet100V3Small
from kimm.utils import make_divisible


class MobileNetV3Test(testing.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        [
            (MobileNet100V3Small.__name__, MobileNet100V3Small),
            (MobileNet100V3Large.__name__, MobileNet100V3Large),
        ]
    )
    def test_mobilenet_v3_base(self, model_class):
        # TODO: test the correctness of the real image
        x = random.uniform([1, 224, 224, 3]) * 255.0
        model = model_class()

        y = model.predict(x)

        self.assertEqual(y.shape, (1, 1000))

    @parameterized.named_parameters(
        [
            (MobileNet100V3Small.__name__, MobileNet100V3Small, 1.0),
            (MobileNet100V3Large.__name__, MobileNet100V3Large, 1.0),
        ]
    )
    def test_mobilenet_v3_feature_extractor(self, model_class, width):
        x = random.uniform([1, 224, 224, 3]) * 255.0
        model = model_class(as_feature_extractor=True)

        y = model.predict(x)

        self.assertIsInstance(y, dict)
        self.assertAllEqual(
            list(y.keys()), model_class.available_feature_keys()
        )
        if "Small" in model_class.__name__:
            self.assertEqual(
                list(y["STEM_S2"].shape),
                [1, 112, 112, make_divisible(16 * width)],
            )
            self.assertEqual(
                list(y["BLOCK0_S4"].shape),
                [1, 56, 56, make_divisible(16 * width)],
            )
            self.assertEqual(
                list(y["BLOCK1_S8"].shape),
                [1, 28, 28, make_divisible(24 * width)],
            )
            self.assertEqual(
                list(y["BLOCK2_S16"].shape),
                [1, 14, 14, make_divisible(40 * width)],
            )
            self.assertEqual(
                list(y["BLOCK4_S32"].shape),
                [1, 7, 7, make_divisible(96 * width)],
            )
        else:
            self.assertEqual(
                list(y["STEM_S2"].shape),
                [1, 112, 112, make_divisible(16 * width)],
            )
            self.assertEqual(
                list(y["BLOCK1_S4"].shape),
                [1, 56, 56, make_divisible(24 * width)],
            )
            self.assertEqual(
                list(y["BLOCK2_S8"].shape),
                [1, 28, 28, make_divisible(40 * width)],
            )
            self.assertEqual(
                list(y["BLOCK3_S16"].shape),
                [1, 14, 14, make_divisible(80 * width)],
            )
            self.assertEqual(
                list(y["BLOCK5_S32"].shape),
                [1, 7, 7, make_divisible(160 * width)],
            )

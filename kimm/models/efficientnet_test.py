from absl.testing import parameterized
from keras import models
from keras import random
from keras.src import testing

from kimm.models.efficientnet import EfficientNetB0
from kimm.models.efficientnet import EfficientNetB2
from kimm.models.efficientnet import EfficientNetLiteB0
from kimm.models.efficientnet import EfficientNetLiteB2
from kimm.models.efficientnet import EfficientNetV2B0
from kimm.models.efficientnet import EfficientNetV2S
from kimm.models.efficientnet import TinyNetE
from kimm.utils import make_divisible


class EfficientNetTest(testing.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        [
            (EfficientNetB0.__name__, EfficientNetB0, 224),
            (EfficientNetB2.__name__, EfficientNetB2, 260),
            (EfficientNetLiteB0.__name__, EfficientNetLiteB0, 224),
            (EfficientNetLiteB2.__name__, EfficientNetLiteB2, 260),
            (EfficientNetV2S.__name__, EfficientNetV2S, 300),
            (EfficientNetV2B0.__name__, EfficientNetV2B0, 192),
            (TinyNetE.__name__, TinyNetE, 106),
        ]
    )
    def test_efficentnet_base(self, model_class, image_size):
        # TODO: test the correctness of the real image
        x = random.uniform([1, image_size, image_size, 3]) * 255.0
        model = model_class()

        y = model(x, training=False)

        self.assertEqual(y.shape, (1, 1000))

    @parameterized.named_parameters(
        [
            (EfficientNetB0.__name__, EfficientNetB0, 1.0, False),
            (EfficientNetB2.__name__, EfficientNetB2, 1.1, False),
            (EfficientNetLiteB0.__name__, EfficientNetLiteB0, 1.0, False),
            (EfficientNetLiteB2.__name__, EfficientNetLiteB2, 1.1, False),
            (TinyNetE.__name__, TinyNetE, 0.51, True),
        ]
    )
    def test_efficentnet_feature_extractor(
        self, model_class, width, fix_stem_channels
    ):
        x = random.uniform([1, 224, 224, 3]) * 255.0
        model = model_class(
            input_shape=[224, 224, 3], as_feature_extractor=True
        )

        y = model(x, training=False)

        self.assertIsInstance(y, dict)
        self.assertAllEqual(
            list(y.keys()), model_class.available_feature_keys()
        )
        if fix_stem_channels:
            self.assertEqual(list(y["STEM_S2"].shape), [1, 112, 112, 32])
        else:
            self.assertEqual(
                list(y["STEM_S2"].shape),
                [1, 112, 112, make_divisible(32 * width)],
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

    @parameterized.named_parameters(
        [
            (EfficientNetV2S.__name__, EfficientNetV2S, 1.0),
            (EfficientNetV2B0.__name__, EfficientNetV2B0, 1.0),
        ]
    )
    def test_efficentnet_v2_feature_extractor(self, model_class, width):
        x = random.uniform([1, 224, 224, 3]) * 255.0
        model = model_class(
            input_shape=[224, 224, 3], as_feature_extractor=True
        )

        y = model(x, training=False)

        self.assertIsInstance(y, dict)
        self.assertAllEqual(
            list(y.keys()), model_class.available_feature_keys()
        )
        if "EfficientNetV2S" in model_class.__name__:
            self.assertEqual(list(y["STEM_S2"].shape), [1, 112, 112, 24])
            self.assertEqual(list(y["BLOCK1_S4"].shape), [1, 56, 56, 48])
            self.assertEqual(list(y["BLOCK2_S8"].shape), [1, 28, 28, 64])
            self.assertEqual(list(y["BLOCK3_S16"].shape), [1, 14, 14, 128])
            self.assertEqual(list(y["BLOCK5_S32"].shape), [1, 7, 7, 256])
        elif "EfficientNetV2B" in model_class.__name__:
            self.assertEqual(
                list(y["STEM_S2"].shape),
                [1, 112, 112, make_divisible(32 * width)],
            )
            self.assertEqual(
                list(y["BLOCK1_S4"].shape),
                [1, 56, 56, make_divisible(32 * width)],
            )
            self.assertEqual(
                list(y["BLOCK2_S8"].shape),
                [1, 28, 28, make_divisible(48 * width)],
            )
            self.assertEqual(
                list(y["BLOCK3_S16"].shape),
                [1, 14, 14, make_divisible(96 * width)],
            )
            self.assertEqual(
                list(y["BLOCK5_S32"].shape),
                [1, 7, 7, make_divisible(192 * width)],
            )

    @parameterized.named_parameters(
        [
            (EfficientNetB0.__name__, EfficientNetB0, 224),
            (EfficientNetLiteB0.__name__, EfficientNetLiteB0, 224),
            (TinyNetE.__name__, TinyNetE, 106),
            (EfficientNetV2S.__name__, EfficientNetV2S, 300),
        ]
    )
    def test_efficientnet_serialization(self, model_class, image_size):
        x = random.uniform([1, image_size, image_size, 3]) * 255.0
        temp_dir = self.get_temp_dir()
        model1 = model_class()
        y1 = model1(x, training=False)
        model1.save(temp_dir + "/model.keras")

        model2 = models.load_model(temp_dir + "/model.keras")
        y2 = model2(x, training=False)

        self.assertAllClose(y1, y2)

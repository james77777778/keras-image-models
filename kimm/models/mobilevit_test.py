import pytest
from absl.testing import parameterized
from keras import models
from keras import random
from keras.src import testing

from kimm.models.mobilevit import MobileViTS
from kimm.models.mobilevit import MobileViTXS


class MobileViTTest(testing.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        [(MobileViTS.__name__, MobileViTS), (MobileViTXS.__name__, MobileViTXS)]
    )
    def test_mobilevit_base(self, model_class):
        # TODO: test the correctness of the real image
        x = random.uniform([1, 256, 256, 3]) * 255.0
        model = model_class()

        y = model(x, training=False)

        self.assertEqual(y.shape, (1, 1000))

    @parameterized.named_parameters(
        [(MobileViTS.__name__, MobileViTS), (MobileViTXS.__name__, MobileViTXS)]
    )
    def test_mobilevit_feature_extractor(self, model_class):
        x = random.uniform([1, 256, 256, 3]) * 255.0
        model = model_class(as_feature_extractor=True)

        y = model(x, training=False)

        self.assertIsInstance(y, dict)
        self.assertAllEqual(
            list(y.keys()), model_class.available_feature_keys()
        )
        if "MobileViTS" in model_class.__name__:
            self.assertEqual(list(y["STEM_S2"].shape), [1, 128, 128, 16])
            self.assertEqual(list(y["BLOCK1_S4"].shape), [1, 64, 64, 64])
            self.assertEqual(list(y["BLOCK2_S8"].shape), [1, 32, 32, 96])
            self.assertEqual(list(y["BLOCK3_S16"].shape), [1, 16, 16, 128])
            self.assertEqual(list(y["BLOCK4_S32"].shape), [1, 8, 8, 160])
        elif "MobileViTXS" in model_class.__name__:
            self.assertEqual(list(y["STEM_S2"].shape), [1, 128, 128, 16])
            self.assertEqual(list(y["BLOCK1_S4"].shape), [1, 64, 64, 48])
            self.assertEqual(list(y["BLOCK2_S8"].shape), [1, 32, 32, 64])
            self.assertEqual(list(y["BLOCK3_S16"].shape), [1, 16, 16, 80])
            self.assertEqual(list(y["BLOCK4_S32"].shape), [1, 8, 8, 96])

    @pytest.mark.serialization
    @parameterized.named_parameters(
        [
            (MobileViTS.__name__, MobileViTS, 256),
            (MobileViTXS.__name__, MobileViTXS, 256),
        ]
    )
    def test_mobilevit_serialization(self, model_class, image_size):
        x = random.uniform([1, image_size, image_size, 3]) * 255.0
        temp_dir = self.get_temp_dir()
        model1 = model_class()
        y1 = model1(x, training=False)
        model1.save(temp_dir + "/model.keras")

        model2 = models.load_model(temp_dir + "/model.keras")
        y2 = model2(x, training=False)

        self.assertAllClose(y1, y2)

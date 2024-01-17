import pytest
from absl.testing import parameterized
from keras import models
from keras import random
from keras.src import testing

from kimm.models.ghostnet import GhostNet100
from kimm.models.ghostnet import GhostNet100V2


class GhostNetTest(testing.TestCase, parameterized.TestCase):
    @parameterized.named_parameters([(GhostNet100.__name__, GhostNet100)])
    def test_ghostnet_base(self, model_class):
        # TODO: test the correctness of the real image
        x = random.uniform([1, 224, 224, 3]) * 255.0
        model = model_class()

        y = model(x, training=False)

        self.assertEqual(y.shape, (1, 1000))

    @parameterized.named_parameters([(GhostNet100.__name__, GhostNet100)])
    def test_ghostnet_feature_extractor(self, model_class):
        x = random.uniform([1, 224, 224, 3]) * 255.0
        model = model_class(as_feature_extractor=True)

        y = model(x, training=False)

        self.assertIsInstance(y, dict)
        self.assertAllEqual(
            list(y.keys()), model_class.available_feature_keys()
        )
        self.assertEqual(list(y["STEM_S2"].shape), [1, 112, 112, 16])
        self.assertEqual(list(y["BLOCK1_S4"].shape), [1, 56, 56, 24])
        self.assertEqual(list(y["BLOCK3_S8"].shape), [1, 28, 28, 40])
        self.assertEqual(list(y["BLOCK5_S16"].shape), [1, 14, 14, 80])
        self.assertEqual(list(y["BLOCK7_S32"].shape), [1, 7, 7, 160])

    @parameterized.named_parameters([(GhostNet100V2.__name__, GhostNet100V2)])
    def test_ghostnetv2_base(self, model_class):
        # TODO: test the correctness of the real image
        x = random.uniform([1, 224, 224, 3]) * 255.0
        model = model_class()

        y = model(x, training=False)

        self.assertEqual(y.shape, (1, 1000))

    @parameterized.named_parameters([(GhostNet100V2.__name__, GhostNet100V2)])
    def test_ghostnetv2_feature_extractor(self, model_class):
        x = random.uniform([1, 224, 224, 3]) * 255.0
        model = model_class(as_feature_extractor=True)

        y = model(x, training=False)

        self.assertIsInstance(y, dict)
        self.assertAllEqual(
            list(y.keys()), model_class.available_feature_keys()
        )
        self.assertEqual(list(y["STEM_S2"].shape), [1, 112, 112, 16])
        self.assertEqual(list(y["BLOCK1_S4"].shape), [1, 56, 56, 24])
        self.assertEqual(list(y["BLOCK3_S8"].shape), [1, 28, 28, 40])
        self.assertEqual(list(y["BLOCK5_S16"].shape), [1, 14, 14, 80])
        self.assertEqual(list(y["BLOCK7_S32"].shape), [1, 7, 7, 160])

    @pytest.mark.serialization
    @parameterized.named_parameters(
        [
            (GhostNet100.__name__, GhostNet100, 224),
            (GhostNet100V2.__name__, GhostNet100V2, 224),
        ]
    )
    def test_ghostnet_serialization(self, model_class, image_size):
        x = random.uniform([1, image_size, image_size, 3]) * 255.0
        temp_dir = self.get_temp_dir()
        model1 = model_class()
        y1 = model1(x, training=False)
        model1.save(temp_dir + "/model.keras")

        model2 = models.load_model(temp_dir + "/model.keras")
        y2 = model2(x, training=False)

        self.assertAllClose(y1, y2)

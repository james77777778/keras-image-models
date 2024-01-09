from absl.testing import parameterized
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

        y = model.predict(x)

        self.assertEqual(y.shape, (1, 1000))

    @parameterized.named_parameters([(GhostNet100.__name__, GhostNet100)])
    def test_ghostnet_feature_extractor(self, model_class):
        x = random.uniform([1, 224, 224, 3]) * 255.0
        model = model_class(as_feature_extractor=True)

        y = model.predict(x)

        self.assertIsInstance(y, dict)
        self.assertEqual(list(y["S2"].shape), [1, 112, 112, 16])
        self.assertEqual(list(y["S4"].shape), [1, 56, 56, 24])
        self.assertEqual(list(y["S8"].shape), [1, 28, 28, 40])
        self.assertEqual(list(y["S16"].shape), [1, 14, 14, 80])
        self.assertEqual(list(y["S32"].shape), [1, 7, 7, 160])

    @parameterized.named_parameters([(GhostNet100V2.__name__, GhostNet100V2)])
    def test_ghostnetv2_base(self, model_class):
        # TODO: test the correctness of the real image
        x = random.uniform([1, 224, 224, 3]) * 255.0
        model = model_class()

        y = model.predict(x)

        self.assertEqual(y.shape, (1, 1000))

    @parameterized.named_parameters([(GhostNet100V2.__name__, GhostNet100V2)])
    def test_ghostnetv2_feature_extractor(self, model_class):
        x = random.uniform([1, 224, 224, 3]) * 255.0
        model = model_class(as_feature_extractor=True)

        y = model.predict(x)

        self.assertIsInstance(y, dict)
        self.assertEqual(list(y["S2"].shape), [1, 112, 112, 16])
        self.assertEqual(list(y["S4"].shape), [1, 56, 56, 24])
        self.assertEqual(list(y["S8"].shape), [1, 28, 28, 40])
        self.assertEqual(list(y["S16"].shape), [1, 14, 14, 80])
        self.assertEqual(list(y["S32"].shape), [1, 7, 7, 160])

from absl.testing import parameterized
from keras import random
from keras.src import testing

from kimm.models.resnet import ResNet18
from kimm.models.resnet import ResNet50


class ResNetTest(testing.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        [(ResNet18.__name__, ResNet18), (ResNet50.__name__, ResNet50)]
    )
    def test_resnet_base(self, model_class):
        # TODO: test the correctness of the real image
        x = random.uniform([1, 224, 224, 3]) * 255.0
        model = model_class()

        y = model.predict(x)

        self.assertEqual(y.shape, (1, 1000))

    @parameterized.named_parameters(
        [(ResNet18.__name__, ResNet18, 1), (ResNet50.__name__, ResNet50, 4)]
    )
    def test_resnet_feature_extractor(self, model_class, expansion):
        x = random.uniform([1, 224, 224, 3]) * 255.0
        model = model_class(as_feature_extractor=True)

        y = model.predict(x)

        self.assertIsInstance(y, dict)
        self.assertEqual(list(y["S2"].shape), [1, 112, 112, 64])
        self.assertEqual(list(y["S4"].shape), [1, 56, 56, 64 * expansion])
        self.assertEqual(list(y["S8"].shape), [1, 28, 28, 128 * expansion])
        self.assertEqual(list(y["S16"].shape), [1, 14, 14, 256 * expansion])
        self.assertEqual(list(y["S32"].shape), [1, 7, 7, 512 * expansion])

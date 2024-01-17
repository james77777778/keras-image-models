import pytest
from absl.testing import parameterized
from keras import models
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

        y = model(x, training=False)

        self.assertEqual(y.shape, (1, 1000))

    @parameterized.named_parameters(
        [(ResNet18.__name__, ResNet18, 1), (ResNet50.__name__, ResNet50, 4)]
    )
    def test_resnet_feature_extractor(self, model_class, expansion):
        x = random.uniform([1, 224, 224, 3]) * 255.0
        model = model_class(as_feature_extractor=True)

        y = model(x, training=False)

        self.assertIsInstance(y, dict)
        self.assertAllEqual(
            list(y.keys()), model_class.available_feature_keys()
        )
        self.assertEqual(list(y["STEM_S2"].shape), [1, 112, 112, 64])
        self.assertEqual(
            list(y["BLOCK0_S4"].shape), [1, 56, 56, 64 * expansion]
        )
        self.assertEqual(
            list(y["BLOCK1_S8"].shape), [1, 28, 28, 128 * expansion]
        )
        self.assertEqual(
            list(y["BLOCK2_S16"].shape), [1, 14, 14, 256 * expansion]
        )
        self.assertEqual(
            list(y["BLOCK3_S32"].shape), [1, 7, 7, 512 * expansion]
        )

    @pytest.mark.serialization
    @parameterized.named_parameters(
        [(ResNet18.__name__, ResNet18, 224), (ResNet50.__name__, ResNet50, 224)]
    )
    def test_resnet_serialization(self, model_class, image_size):
        x = random.uniform([1, image_size, image_size, 3]) * 255.0
        temp_dir = self.get_temp_dir()
        model1 = model_class()
        y1 = model1(x, training=False)
        model1.save(temp_dir + "/model.keras")

        model2 = models.load_model(temp_dir + "/model.keras")
        y2 = model2(x, training=False)

        self.assertAllClose(y1, y2)

import pytest
from absl.testing import parameterized
from keras import models
from keras import random
from keras.src import testing

from kimm.models.densenet import DenseNet121


class DenseNetTest(testing.TestCase, parameterized.TestCase):
    @parameterized.named_parameters([(DenseNet121.__name__, DenseNet121)])
    def test_densenet_base(self, model_class):
        # TODO: test the correctness of the real image
        x = random.uniform([1, 224, 224, 3]) * 255.0
        model = model_class(input_shape=[224, 224, 3])

        y = model(x, training=False)

        self.assertEqual(y.shape, (1, 1000))

    @parameterized.named_parameters([(DenseNet121.__name__, DenseNet121)])
    def test_densenet_feature_extractor(self, model_class):
        x = random.uniform([1, 224, 224, 3]) * 255.0
        model = model_class(input_shape=[224, 224, 3], feature_extractor=True)

        y = model(x, training=False)

        self.assertIsInstance(y, dict)
        self.assertContainsSubset(
            model_class.available_feature_keys(),
            list(y.keys()),
        )
        self.assertEqual(list(y["STEM_S4"].shape), [1, 56, 56, 64])
        self.assertEqual(list(y["BLOCK0_S8"].shape), [1, 28, 28, 128])
        self.assertEqual(list(y["BLOCK1_S16"].shape), [1, 14, 14, 256])
        self.assertEqual(list(y["BLOCK2_S32"].shape), [1, 7, 7, 512])
        self.assertEqual(list(y["BLOCK3_S32"].shape), [1, 7, 7, 1024])

    @pytest.mark.serialization
    @parameterized.named_parameters([(DenseNet121.__name__, DenseNet121, 224)])
    def test_densenet_serialization(self, model_class, image_size):
        x = random.uniform([1, image_size, image_size, 3]) * 255.0
        temp_dir = self.get_temp_dir()
        model1 = model_class(input_shape=[224, 224, 3])
        y1 = model1(x, training=False)
        model1.save(temp_dir + "/model.keras")

        model2 = models.load_model(temp_dir + "/model.keras")
        y2 = model2(x, training=False)

        self.assertAllClose(y1, y2)

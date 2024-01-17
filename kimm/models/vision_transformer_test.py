import pytest
from absl.testing import parameterized
from keras import models
from keras import random
from keras.src import testing

from kimm.models.vision_transformer import VisionTransformerTiny16
from kimm.models.vision_transformer import VisionTransformerTiny32


class VisionTransformerTest(testing.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        [
            (VisionTransformerTiny16.__name__, VisionTransformerTiny16),
            (VisionTransformerTiny32.__name__, VisionTransformerTiny32),
        ]
    )
    def test_vision_transformer_base(self, model_class):
        # TODO: test the correctness of the real image
        x = random.uniform([1, 384, 384, 3]) * 255.0
        model = model_class()

        y = model(x, training=False)

        self.assertEqual(y.shape, (1, 1000))

    @parameterized.named_parameters(
        [
            (VisionTransformerTiny16.__name__, VisionTransformerTiny16, 16),
            (VisionTransformerTiny32.__name__, VisionTransformerTiny32, 32),
        ]
    )
    def test_vision_transformer_feature_extractor(
        self, model_class, patch_size
    ):
        x = random.uniform([1, 384, 384, 3]) * 255.0
        model = model_class(as_feature_extractor=True)

        y = model(x, training=False)

        self.assertIsInstance(y, dict)
        self.assertAllEqual(
            list(y.keys()), model_class.available_feature_keys()
        )
        if patch_size == 16:
            self.assertEqual(list(y["BLOCK0"].shape), [1, 577, 192])
        elif patch_size == 32:
            self.assertEqual(list(y["BLOCK0"].shape), [1, 145, 192])
        if patch_size == 16:
            self.assertEqual(list(y["BLOCK5"].shape), [1, 577, 192])
        elif patch_size == 32:
            self.assertEqual(list(y["BLOCK5"].shape), [1, 145, 192])

    @pytest.mark.serialization
    @parameterized.named_parameters(
        [
            (VisionTransformerTiny16.__name__, VisionTransformerTiny16, 384),
            (VisionTransformerTiny32.__name__, VisionTransformerTiny32, 384),
        ]
    )
    def test_vit_serialization(self, model_class, image_size):
        x = random.uniform([1, image_size, image_size, 3]) * 255.0
        temp_dir = self.get_temp_dir()
        model1 = model_class()
        y1 = model1(x, training=False)
        model1.save(temp_dir + "/model.keras")

        model2 = models.load_model(temp_dir + "/model.keras")
        y2 = model2(x, training=False)

        self.assertAllClose(y1, y2)

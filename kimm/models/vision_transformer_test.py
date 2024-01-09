from absl.testing import parameterized
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

        y = model.predict(x)

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

        y = model.predict(x)

        self.assertIsInstance(y, dict)
        if patch_size == 16:
            self.assertEqual(list(y["Depth0"].shape), [1, 577, 192])
        elif patch_size == 32:
            self.assertEqual(list(y["Depth0"].shape), [1, 145, 192])
        if patch_size == 16:
            self.assertEqual(list(y["Depth5"].shape), [1, 577, 192])
        elif patch_size == 32:
            self.assertEqual(list(y["Depth5"].shape), [1, 145, 192])

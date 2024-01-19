from absl.testing import parameterized
from keras import layers
from keras import random
from keras.src import testing

from kimm.models.base_model import BaseModel


class SampleModel(BaseModel):
    def __init__(self, **kwargs):
        self.set_properties(kwargs)
        inputs = layers.Input(shape=[224, 224, 3])

        features = {}
        s2 = layers.Conv2D(3, 1, 2, use_bias=False)(inputs)
        features["S2"] = s2
        s4 = layers.Conv2D(3, 1, 2, use_bias=False)(s2)
        features["S4"] = s4
        s8 = layers.Conv2D(3, 1, 2, use_bias=False)(s4)
        features["S8"] = s8
        s16 = layers.Conv2D(3, 1, 2, use_bias=False)(s8)
        features["S16"] = s16
        s32 = layers.Conv2D(3, 1, 2, use_bias=False)(s16)
        features["S32"] = s32
        super().__init__(
            inputs=inputs, outputs=s32, features=features, **kwargs
        )

    @staticmethod
    def available_feature_keys():
        # predefined for better UX
        return [f"S{2**i}" for i in range(1, 6)]

    def get_config(self):
        return super().get_config()


class BaseModelTest(testing.TestCase, parameterized.TestCase):
    def test_feature_extractor(self):
        x = random.uniform([1, 224, 224, 3])

        # availiable_feature_keys
        self.assertContainsSubset(
            ["S2", "S4", "S8", "S16", "S32"],
            SampleModel.available_feature_keys(),
        )

        # feature_extractor=False
        model = SampleModel()

        y = model(x, training=False)

        self.assertNotIsInstance(y, dict)
        self.assertEqual(list(y.shape), [1, 7, 7, 3])

        # feature_extractor=True
        model = SampleModel(feature_extractor=True)

        y = model(x, training=False)

        self.assertIsInstance(y, dict)
        self.assertEqual(list(y["S2"].shape), [1, 112, 112, 3])
        self.assertEqual(list(y["S32"].shape), [1, 7, 7, 3])

        # feature_extractor=True with feature_keys
        model = SampleModel(
            feature_extractor=True, feature_keys=["S2", "S16", "S32"]
        )

        y = model(x, training=False)

        self.assertIsInstance(y, dict)
        self.assertNotIn("S4", y)
        self.assertNotIn("S8", y)
        self.assertEqual(list(y["S2"].shape), [1, 112, 112, 3])
        self.assertEqual(list(y["S16"].shape), [1, 14, 14, 3])
        self.assertEqual(list(y["S32"].shape), [1, 7, 7, 3])

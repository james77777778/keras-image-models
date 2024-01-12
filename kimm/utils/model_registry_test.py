from keras import models
from keras.src import testing

from kimm.models.feature_extractor import FeatureExtractor
from kimm.utils.model_registry import MODEL_REGISTRY
from kimm.utils.model_registry import add_model_to_registry
from kimm.utils.model_registry import clear_registry
from kimm.utils.model_registry import list_models


class DummyModel(models.Model):
    pass


class DummyFeatureExtractor(FeatureExtractor):
    @staticmethod
    def available_feature_keys():
        return ["A", "B", "C"]


class ModelRegistryTest(testing.TestCase):
    def test_add_model_to_registry(self):
        clear_registry()
        self.assertEqual(len(MODEL_REGISTRY), 0)

        add_model_to_registry(DummyModel, False)
        self.assertEqual(len(MODEL_REGISTRY), 1)
        self.assertEqual(MODEL_REGISTRY[0]["name"], DummyModel.__name__)
        self.assertEqual(MODEL_REGISTRY[0]["support_feature"], False)
        self.assertEqual(MODEL_REGISTRY[0]["available_feature_keys"], [])
        self.assertEqual(MODEL_REGISTRY[0]["has_pretrained"], False)

        add_model_to_registry(DummyFeatureExtractor, True)
        self.assertEqual(len(MODEL_REGISTRY), 2)
        self.assertEqual(
            MODEL_REGISTRY[1]["name"], DummyFeatureExtractor.__name__
        )
        self.assertEqual(MODEL_REGISTRY[1]["support_feature"], True)
        self.assertEqual(
            MODEL_REGISTRY[1]["available_feature_keys"], ["A", "B", "C"]
        )
        self.assertEqual(MODEL_REGISTRY[1]["has_pretrained"], True)

    def test_add_model_to_registry_invalid(self):
        clear_registry()
        add_model_to_registry(DummyModel, False)
        with self.assertRaisesRegex(
            ValueError, "MODEL_REGISTRY already contains"
        ):
            add_model_to_registry(DummyModel, False)

    def test_list_models(self):
        clear_registry()
        add_model_to_registry(DummyModel, False)
        add_model_to_registry(DummyFeatureExtractor, True)

        # all models
        result = list_models()
        self.assertEqual(len(result), 2)
        self.assertTrue(DummyModel.__name__ in result)
        self.assertTrue(DummyFeatureExtractor.__name__ in result)

        # filter name
        result = list_models("DummyModel")
        self.assertEqual(len(result), 1)
        self.assertTrue(DummyModel.__name__ in result)
        self.assertTrue(DummyFeatureExtractor.__name__ not in result)

        # filter support_feature
        result = list_models(support_feature=True)
        self.assertEqual(len(result), 1)
        self.assertTrue(DummyModel.__name__ not in result)
        self.assertTrue(DummyFeatureExtractor.__name__ in result)

        # filter pretrained
        result = list_models(has_pretrained=True)
        self.assertEqual(len(result), 1)
        self.assertTrue(DummyModel.__name__ not in result)
        self.assertTrue(DummyFeatureExtractor.__name__ in result)

        # filter multiple conditions
        result = list_models(support_feature=True, has_pretrained=False)
        self.assertEqual(len(result), 0)
        self.assertTrue(DummyModel.__name__ not in result)
        self.assertTrue(DummyFeatureExtractor.__name__ not in result)

        result = list_models("Dummy", support_feature=True, has_pretrained=True)
        self.assertEqual(len(result), 1)
        self.assertTrue(DummyModel.__name__ not in result)
        self.assertTrue(DummyFeatureExtractor.__name__ in result)

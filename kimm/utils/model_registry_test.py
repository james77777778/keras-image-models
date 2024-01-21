from keras import models
from keras.src import testing

from kimm.models.base_model import BaseModel
from kimm.utils.model_registry import MODEL_REGISTRY
from kimm.utils.model_registry import add_model_to_registry
from kimm.utils.model_registry import clear_registry
from kimm.utils.model_registry import list_models


class DummyModel(models.Model):
    pass


class DummyFeatureExtractor(BaseModel):
    available_feature_keys = ["A", "B", "C"]


class ModelRegistryTest(testing.TestCase):
    def test_add_model_to_registry(self):
        clear_registry()
        self.assertEqual(len(MODEL_REGISTRY), 0)

        add_model_to_registry(DummyModel, None)
        self.assertEqual(len(MODEL_REGISTRY), 1)
        self.assertEqual(MODEL_REGISTRY[0]["name"], DummyModel.__name__)
        self.assertEqual(MODEL_REGISTRY[0]["feature_extractor"], False)
        self.assertEqual(MODEL_REGISTRY[0]["feature_keys"], [])
        self.assertEqual(MODEL_REGISTRY[0]["weights"], None)

        add_model_to_registry(DummyFeatureExtractor, "imagenet")
        self.assertEqual(len(MODEL_REGISTRY), 2)
        self.assertEqual(
            MODEL_REGISTRY[1]["name"], DummyFeatureExtractor.__name__
        )
        self.assertEqual(MODEL_REGISTRY[1]["feature_extractor"], True)
        self.assertEqual(MODEL_REGISTRY[1]["feature_keys"], ["A", "B", "C"])
        self.assertEqual(MODEL_REGISTRY[1]["weights"], "imagenet")

    def test_add_model_to_registry_invalid(self):
        clear_registry()
        add_model_to_registry(DummyModel, None)
        with self.assertWarnsRegex(Warning, "MODEL_REGISTRY already contains"):
            add_model_to_registry(DummyModel, None)

    def test_list_models(self):
        clear_registry()
        add_model_to_registry(DummyModel, None)
        add_model_to_registry(DummyFeatureExtractor, "imagenet")

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

        # filter feature_extractor
        result = list_models(feature_extractor=True)
        self.assertEqual(len(result), 1)
        self.assertTrue(DummyModel.__name__ not in result)
        self.assertTrue(DummyFeatureExtractor.__name__ in result)

        # filter weights="imagenet"
        result = list_models(weights="imagenet")
        self.assertEqual(len(result), 1)
        self.assertTrue(DummyModel.__name__ not in result)
        self.assertTrue(DummyFeatureExtractor.__name__ in result)

        # filter weights=True
        result = list_models(weights=True)
        self.assertEqual(len(result), 1)
        self.assertTrue(DummyModel.__name__ not in result)
        self.assertTrue(DummyFeatureExtractor.__name__ in result)

        # filter multiple conditions
        result = list_models(feature_extractor=True, weights=False)
        self.assertEqual(len(result), 0)
        self.assertTrue(DummyModel.__name__ not in result)
        self.assertTrue(DummyFeatureExtractor.__name__ not in result)

        result = list_models(
            "Dummy", feature_extractor=True, weights="imagenet"
        )
        self.assertEqual(len(result), 1)
        self.assertTrue(DummyModel.__name__ not in result)
        self.assertTrue(DummyFeatureExtractor.__name__ in result)

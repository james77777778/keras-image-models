import pytest
from absl.testing import parameterized
from keras import backend
from keras import ops
from keras import random
from keras.src import testing

from kimm._src import models
from kimm._src.export import export_tflite


class ExportTFLiteTest(testing.TestCase, parameterized.TestCase):
    def get_model_and_representative_dataset(self):
        input_shape = [224, 224, 3]
        model = models.mobilenet_v3.MobileNetV3W050Small(
            include_preprocessing=False, weights=None
        )

        def representative_dataset():
            for _ in range(10):
                yield [
                    ops.convert_to_numpy(
                        random.uniform([1, *input_shape], maxval=255.0)
                    )
                ]

        return input_shape, model, representative_dataset

    @classmethod
    def setUpClass(cls):
        cls.original_image_data_format = backend.image_data_format()

    @classmethod
    def tearDownClass(cls):
        backend.set_image_data_format(cls.original_image_data_format)

    @pytest.mark.skipif(
        backend.backend() != "tensorflow", reason="Requires tensorflow backend."
    )
    def test_export_tflite_fp32(self):
        (input_shape, model, _) = self.get_model_and_representative_dataset()
        temp_dir = self.get_temp_dir()

        export_tflite.export_tflite(
            model, input_shape, f"{temp_dir}/model_fp32.onnx", "float32"
        )

    @pytest.mark.skipif(
        backend.backend() != "tensorflow", reason="Requires tensorflow backend."
    )
    def test_export_tflite_fp16(self):
        (input_shape, model, _) = self.get_model_and_representative_dataset()
        temp_dir = self.get_temp_dir()

        export_tflite.export_tflite(
            model, input_shape, f"{temp_dir}/model_fp16.tflite", "float16"
        )

    @pytest.mark.skipif(
        backend.backend() != "tensorflow", reason="Requires tensorflow backend."
    )
    def test_export_tflite_int8(self):
        (
            input_shape,
            model,
            representative_dataset,
        ) = self.get_model_and_representative_dataset()
        temp_dir = self.get_temp_dir()

        export_tflite.export_tflite(
            model,
            input_shape,
            f"{temp_dir}/model_int8.tflite",
            "int8",
            representative_dataset,
        )

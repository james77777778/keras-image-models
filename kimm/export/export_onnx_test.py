import pytest
from absl.testing import parameterized
from keras import backend
from keras.src import testing

from kimm import export
from kimm import models


class ExportOnnxTest(testing.TestCase, parameterized.TestCase):
    def get_model(self):
        input_shape = [3, 224, 224]  # channels_first
        model = models.MobileNetV3W050Small(include_preprocessing=False)
        return input_shape, model

    @classmethod
    def setUpClass(cls):
        cls.original_image_data_format = backend.image_data_format()

    @classmethod
    def tearDownClass(cls):
        backend.set_image_data_format(cls.original_image_data_format)

    @pytest.mark.skipif(
        backend.backend() != "torch", reason="Requires torch backend."
    )
    def DISABLE_test_export_onnx_use(self):
        # TODO: turn on this test
        # SystemError: <method '__int__' of 'torch._C._TensorBase' objects>
        # returned a result with an exception set
        backend.set_image_data_format("channels_first")
        input_shape, model = self.get_model()

        temp_dir = self.get_temp_dir()

        export.export_onnx(model, input_shape, f"{temp_dir}/model.onnx")

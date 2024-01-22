import pytest
from absl.testing import parameterized
from keras import backend
from keras.src import testing

from kimm import export
from kimm import models


class ExportOnnxTest(testing.TestCase, parameterized.TestCase):
    def get_model(self):
        input_shape = [224, 224, 3]
        model = models.MobileNet050V3Small(include_preprocessing=False)
        return input_shape, model

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",  # TODO: test torch
        reason="Requires tensorflow or torch backend.",
    )
    def test_export_onnx_use(self):
        input_shape, model = self.get_model()

        temp_dir = self.get_temp_dir()

        if backend.backend() == "tensorflow":
            export.export_onnx(model, input_shape, f"{temp_dir}/model.onnx")
        elif backend.backend() == "torch":
            export.export_onnx(
                model, input_shape, f"{temp_dir}/model.onnx", use_nchw=False
            )

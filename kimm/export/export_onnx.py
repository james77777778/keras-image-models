import pathlib
import typing

from keras import backend
from keras import layers
from keras import models
from keras import ops

from kimm.models import BaseModel
from kimm.utils.module_utils import torch


def export_onnx(
    model: BaseModel,
    input_shape: typing.Union[int, typing.Sequence[int]],
    export_path: typing.Union[str, pathlib.Path],
    batch_size: int = 1,
):
    """Export the model to onnx format (in float32).

    Only torch backend with 'channels_first' is supported. The onnx model will
    be generated using `torch.onnx.export` and optimized through `onnxsim` and
    `onnxoptimizer`.

    Note that `onnx`, `onnxruntime`, `onnxsim` and `onnxoptimizer` must be
    installed.

    Args:
        model: keras.Model, the model to be exported.
        input_shape: int or sequence of int, specifying the shape of the input.
        export_path: str or pathlib.Path, specifying the path to export.
        batch_size: int, specifying the batch size of the input,
            defaults to `1`.
    """
    if backend.backend() != "torch":
        raise ValueError("`export_onnx` only supports torch backend")
    if backend.image_data_format() != "channels_first":
        raise ValueError(
            "`export_onnx` only supports 'channels_first' data format."
        )
    try:
        import onnx
        import onnxoptimizer
        import onnxsim
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Failed to import 'onnx', 'onnxsim' or 'onnxoptimizer'. "
            "Please install them by the following instruction:\n"
            "'pip install torch onnx onnxsim onnxoptimizer'"
        )

    if isinstance(input_shape, int):
        input_shape = [3, input_shape, input_shape]
    elif len(input_shape) == 2:
        input_shape = [3, input_shape[0], input_shape[1]]
    elif len(input_shape) == 3:
        input_shape = input_shape

    # Fix input shape
    inputs = layers.Input(
        shape=input_shape, batch_size=batch_size, name="inputs"
    )
    outputs = model(inputs, training=False)
    model = models.Model(inputs, outputs)
    model = model.eval()

    full_input_shape = [1] + list(input_shape)
    dummy_inputs = ops.ones(full_input_shape, dtype="float32")
    scripted_model = torch.jit.trace(
        model.forward, example_inputs=[dummy_inputs]
    )
    torch.onnx.export(scripted_model, dummy_inputs, export_path)

    # Further optimization
    model = onnx.load(export_path)
    model_simp, _ = onnxsim.simplify(model)
    model_simp = onnxoptimizer.optimize(model_simp)
    onnx.save(model_simp, export_path)

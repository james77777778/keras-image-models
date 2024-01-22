import pathlib
import tempfile
import typing

from keras import backend
from keras import layers
from keras import models
from keras import ops

from kimm.models import BaseModel


def _export_onnx_tf(
    model: BaseModel,
    inputs_as_nchw,
    export_path: typing.Union[str, pathlib.Path],
):
    try:
        import tf2onnx
        import tf2onnx.tf_loader
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Failed to import 'tf2onnx'. Please install it by the following "
            "instruction:\n'pip install tf2onnx'"
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir, "temp_saved_model")
        model.export(temp_path)

        (
            graph_def,
            inputs,
            outputs,
            tensors_to_rename,
        ) = tf2onnx.tf_loader.from_saved_model(
            temp_path,
            None,
            None,
            return_tensors_to_rename=True,
        )

        tf2onnx.convert.from_graph_def(
            graph_def,
            input_names=inputs,
            output_names=outputs,
            output_path=export_path,
            inputs_as_nchw=inputs_as_nchw,
            tensors_to_rename=tensors_to_rename,
        )


def _export_onnx_torch(
    model: BaseModel,
    input_shape: typing.Union[int, int, int],
    export_path: typing.Union[str, pathlib.Path],
):
    try:
        import torch
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Failed to import 'torch'. Please install it before calling"
            "`export_onnx` using torch backend"
        )
    full_input_shape = [1] + list(input_shape)
    dummy_inputs = ops.ones(full_input_shape)
    scripted_model = torch.jit.trace(model, dummy_inputs).eval()
    torch.onnx.export(scripted_model, dummy_inputs, export_path)


def export_onnx(
    model: BaseModel,
    input_shape: typing.Union[int, typing.Sequence[int]],
    export_path: typing.Union[str, pathlib.Path],
    batch_size: int = 1,
    use_nchw: bool = True,
):
    if backend.backend() not in ("tensorflow", "torch"):
        raise ValueError(
            "Currently, `export_onnx` only supports tensorflow and torch "
            "backend"
        )
    try:
        import onnx
        import onnxoptimizer
        import onnxsim
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Failed to import 'onnx', 'onnxsim' or 'onnxoptimizer'. Please "
            "install them by the following instruction:\n"
            "'pip install onnx onnxsim onnxoptimizer'"
        )

    if isinstance(input_shape, int):
        input_shape = [input_shape, input_shape, 3]
    elif len(input_shape) == 2:
        input_shape = [input_shape[0], input_shape[1], 3]
    elif len(input_shape) == 3:
        input_shape = input_shape
    if use_nchw:
        if backend.backend() == "torch":
            raise ValueError(
                "Currently, torch backend doesn't support `use_nchw=True`. "
                "You can use tensorflow backend to overcome this issue or "
                "set `use_nchw=False`. "
                "Note that there might be a significant performance "
                "degradation when using torch backend to export onnx due to "
                "the pre- and post-transpose of the Conv2D."
            )
        elif backend.backend() == "tensorflow":
            inputs_as_nchw = ["inputs"]
        else:
            inputs_as_nchw = None
    else:
        inputs_as_nchw = None

    # Fix input shape
    inputs = layers.Input(
        shape=input_shape, batch_size=batch_size, name="inputs"
    )
    outputs = model(inputs, training=False)
    model = models.Model(inputs, outputs)

    if backend.backend() == "tensorflow":
        _export_onnx_tf(model, inputs_as_nchw, export_path)
    elif backend.backend() == "torch":
        _export_onnx_torch(model, input_shape, export_path)

    # Further optimization
    model = onnx.load(export_path)
    model_simp, _ = onnxsim.simplify(model)
    model_simp = onnxoptimizer.optimize(model_simp)
    onnx.save(model_simp, export_path)

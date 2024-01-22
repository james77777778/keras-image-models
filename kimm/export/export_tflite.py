import pathlib
import tempfile
import typing

from keras import backend
from keras import layers
from keras import models
from keras.src.utils.module_utils import tensorflow as tf

from kimm.models import BaseModel


def export_tflite(
    model: BaseModel,
    input_shape: typing.Union[int, typing.Sequence[int]],
    export_path: typing.Union[str, pathlib.Path],
    export_dtype: typing.Literal["float32", "float16", "int8"] = "float32",
    representative_dataset: typing.Optional[typing.Iterator] = None,
    batch_size: int = 1,
):
    if backend.backend() != "tensorflow":
        raise ValueError(
            "Currently, `export_tflite` only supports tensorflow backend"
        )
    if export_dtype not in ("float32", "float16", "int8"):
        raise ValueError(
            "`export_dtype` must be one of ('float32', 'float16', 'int8'). "
            f"Received: export_dtype={export_dtype}"
        )
    if export_dtype == "int8" and representative_dataset is None:
        raise ValueError(
            "For full integer quantization, a `representative_dataset` should "
            "be specified."
        )
    if isinstance(input_shape, int):
        input_shape = [input_shape, input_shape, 3]
    elif len(input_shape) == 2:
        input_shape = [input_shape[0], input_shape[1], 3]
    elif len(input_shape) == 3:
        input_shape = input_shape

    # Fix input shape
    inputs = layers.Input(shape=input_shape, batch_size=batch_size)
    outputs = model(inputs, training=False)
    model = models.Model(inputs, outputs)

    # Construct TFLiteConverter
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir, "temp_saved_model")
        model.export(temp_path)
        converter = tf.lite.TFLiteConverter.from_saved_model(str(temp_path))

        # Configure converter
        if export_dtype != "float32":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if export_dtype == "int8":
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        elif export_dtype == "float16":
            converter.target_spec.supported_types = [tf.float16]
        if representative_dataset is not None:
            converter.representative_dataset = representative_dataset

        # Convert
        tflite_model = converter.convert()

    # Export
    with open(export_path, "wb") as f:
        f.write(tflite_model)

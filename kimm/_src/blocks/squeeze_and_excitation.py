import typing

from keras import backend
from keras import layers

from kimm._src.kimm_export import kimm_export
from kimm._src.utils.make_divisble import make_divisible


@kimm_export(parent_path=["kimm.blocks"])
def apply_se_block(
    inputs,
    se_ratio: float = 0.25,
    activation: typing.Optional[str] = "relu",
    gate_activation: typing.Optional[str] = "sigmoid",
    make_divisible_number: typing.Optional[int] = None,
    se_input_channels: typing.Optional[int] = None,
    name: str = "se_block",
):
    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3
    input_channels = inputs.shape[channels_axis]
    if se_input_channels is None:
        se_input_channels = input_channels
    if make_divisible_number is None:
        se_channels = round(se_input_channels * se_ratio)
    else:
        se_channels = make_divisible(
            se_input_channels * se_ratio, make_divisible_number
        )

    x = inputs
    x = layers.GlobalAveragePooling2D(
        data_format=backend.image_data_format(),
        keepdims=True,
        name=f"{name}_mean",
    )(x)
    x = layers.Conv2D(
        se_channels, 1, use_bias=True, name=f"{name}_conv_reduce"
    )(x)
    if activation is not None:
        x = layers.Activation(activation, name=f"{name}_act1")(x)
    x = layers.Conv2D(
        input_channels, 1, use_bias=True, name=f"{name}_conv_expand"
    )(x)
    if activation is not None:
        x = layers.Activation(gate_activation, name=f"{name}_gate")(x)
    x = layers.Multiply(name=name)([inputs, x])
    return x

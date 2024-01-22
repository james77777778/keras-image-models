import typing

from keras import backend
from keras import layers

from kimm.utils import make_divisible


def apply_activation(
    inputs, activation: typing.Optional[str] = None, name: str = "activation"
):
    x = inputs
    if activation is not None:
        if isinstance(activation, str):
            x = layers.Activation(activation, name=name)(x)
        elif isinstance(activation, layers.Layer):
            x = activation(x)
        else:
            NotImplementedError(
                f"Unsupported activation type: {type(activation)}"
            )
    return x


def apply_conv2d_block(
    inputs,
    filters: typing.Optional[int] = None,
    kernel_size: typing.Optional[
        typing.Union[int, typing.Sequence[int]]
    ] = None,
    strides: int = 1,
    groups: int = 1,
    activation: typing.Optional[str] = None,
    use_depthwise: bool = False,
    add_skip: bool = False,
    bn_momentum: float = 0.9,
    bn_epsilon: float = 1e-5,
    padding: typing.Optional[typing.Literal["same", "valid"]] = None,
    name="conv2d_block",
):
    if kernel_size is None:
        raise ValueError(
            f"kernel_size must be passed. Received: kernel_size={kernel_size}"
        )
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]

    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3
    input_channels = inputs.shape[channels_axis]
    has_skip = add_skip and strides == 1 and input_channels == filters
    x = inputs

    if padding is None:
        padding = "same"
        if strides > 1:
            padding = "valid"
            x = layers.ZeroPadding2D(
                (kernel_size[0] // 2, kernel_size[1] // 2), name=f"{name}_pad"
            )(x)

    if not use_depthwise:
        x = layers.Conv2D(
            filters,
            kernel_size,
            strides,
            padding=padding,
            groups=groups,
            use_bias=False,
            name=f"{name}_conv2d",
        )(x)
    else:
        x = layers.DepthwiseConv2D(
            kernel_size,
            strides,
            padding=padding,
            use_bias=False,
            name=f"{name}_dwconv2d",
        )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        name=f"{name}_bn",
        momentum=bn_momentum,
        epsilon=bn_epsilon,
    )(x)
    x = apply_activation(x, activation, name=name)
    if has_skip:
        x = layers.Add()([x, inputs])
    return x


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
    x = apply_activation(x, activation, name=f"{name}_act1")
    x = layers.Conv2D(
        input_channels, 1, use_bias=True, name=f"{name}_conv_expand"
    )(x)
    x = apply_activation(x, gate_activation, name=f"{name}_gate")
    x = layers.Multiply(name=name)([inputs, x])
    return x

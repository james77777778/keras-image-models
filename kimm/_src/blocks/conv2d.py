import typing

from keras import backend
from keras import layers

from kimm._src.kimm_export import kimm_export


@kimm_export(parent_path=["kimm.blocks"])
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
                ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2),
                name=f"{name}_pad",
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
    if activation is not None:
        x = layers.Activation(activation, name=name)(x)
    if has_skip:
        x = layers.Add()([x, inputs])
    return x

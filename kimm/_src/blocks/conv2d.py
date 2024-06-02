import typing

from keras import backend
from keras import layers
from keras.src.utils.argument_validation import standardize_tuple

from kimm._src.kimm_export import kimm_export


@kimm_export(parent_path=["kimm.blocks"])
def apply_conv2d_block(
    inputs,
    filters: typing.Optional[int] = None,
    kernel_size: typing.Union[int, typing.Sequence[int]] = 1,
    strides: int = 1,
    groups: int = 1,
    activation: typing.Optional[str] = None,
    use_depthwise: bool = False,
    has_skip: bool = False,
    bn_momentum: float = 0.9,
    bn_epsilon: float = 1e-5,
    padding: typing.Optional[typing.Literal["same", "valid"]] = None,
    name="conv2d_block",
):
    """(ZeroPadding) + Conv2D/DepthwiseConv2D + BN + (Activation)."""
    if kernel_size is None:
        raise ValueError(
            f"kernel_size must be passed. Received: kernel_size={kernel_size}"
        )
    kernel_size = standardize_tuple(kernel_size, 2, "kernel_size")

    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3
    input_filters = inputs.shape[channels_axis]
    if has_skip and (strides != 1 or input_filters != filters):
        raise ValueError(
            "If `has_skip=True`, strides must be 1 and `filters` must be the "
            "same as input_filters. "
            f"Received: strides={strides}, filters={filters}, "
            f"input_filters={input_filters}"
        )
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

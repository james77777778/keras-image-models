from keras import layers

from kimm.utils import make_divisible


def apply_activation(x, activation=None, name="activation"):
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
    filters=None,
    kernel_size=None,
    strides=1,
    groups=1,
    activation=None,
    use_depthwise=False,
    bn_momentum=0.9,
    bn_epsilon=1e-5,
    name="conv2d_block",
):
    if kernel_size is None:
        raise ValueError(
            f"kernel_size must be passed. Received: kernel_size={kernel_size}"
        )
    x = inputs

    padding = "same"
    if strides > 1:
        padding = "valid"
        x = layers.ZeroPadding2D(kernel_size // 2, name=f"{name}_pad")(x)

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
        name=f"{name}_bn", momentum=bn_momentum, epsilon=bn_epsilon
    )(x)
    x = apply_activation(x, activation, name=name)
    return x


def apply_se_block(
    inputs,
    se_ratio=0.25,
    activation="relu",
    gate_activation="sigmoid",
    make_divisible_number=None,
    name="se_block",
):
    input_channels = inputs.shape[-1]
    if make_divisible_number is None:
        se_channels = round(input_channels * se_ratio)
    else:
        se_channels = make_divisible(
            input_channels * se_ratio, make_divisible_number
        )

    ori_x = inputs
    x = inputs
    x = layers.GlobalAveragePooling2D(keepdims=True, name=f"{name}_mean")(x)
    x = layers.Conv2D(
        se_channels, 1, use_bias=True, name=f"{name}_reduce_conv2d"
    )(x)
    x = apply_activation(x, activation, name=f"{name}_act")
    x = layers.Conv2D(
        input_channels, 1, use_bias=True, name=f"{name}_expand_conv2d"
    )(x)
    x = apply_activation(x, gate_activation, name=f"{name}_gate_act")
    out = layers.Multiply(name=name)([ori_x, x])
    return out

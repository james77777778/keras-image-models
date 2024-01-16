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
    add_skip=False,
    bn_momentum=0.9,
    bn_epsilon=1e-5,
    padding=None,
    name="conv2d_block",
):
    if kernel_size is None:
        raise ValueError(
            f"kernel_size must be passed. Received: kernel_size={kernel_size}"
        )
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    input_channels = inputs.shape[-1]
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
        name=f"{name}_bn", momentum=bn_momentum, epsilon=bn_epsilon
    )(x)
    x = apply_activation(x, activation, name=name)
    if has_skip:
        x = layers.Add()([x, inputs])
    return x


def apply_se_block(
    inputs,
    se_ratio=0.25,
    activation="relu",
    gate_activation="sigmoid",
    make_divisible_number=None,
    se_input_channels=None,
    name="se_block",
):
    input_channels = inputs.shape[-1]
    if se_input_channels is None:
        se_input_channels = input_channels
    if make_divisible_number is None:
        se_channels = round(se_input_channels * se_ratio)
    else:
        se_channels = make_divisible(
            se_input_channels * se_ratio, make_divisible_number
        )

    ori_x = inputs
    x = inputs
    x = layers.GlobalAveragePooling2D(keepdims=True, name=f"{name}_mean")(x)
    x = layers.Conv2D(
        se_channels, 1, use_bias=True, name=f"{name}_conv_reduce"
    )(x)
    x = apply_activation(x, activation, name=f"{name}_act1")
    x = layers.Conv2D(
        input_channels, 1, use_bias=True, name=f"{name}_conv_expand"
    )(x)
    x = apply_activation(x, gate_activation, name=f"{name}_gate")
    out = layers.Multiply(name=name)([ori_x, x])
    return out

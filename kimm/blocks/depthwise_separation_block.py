from keras import layers

from kimm.blocks.base_block import apply_conv2d_block
from kimm.blocks.base_block import apply_se_block


def apply_depthwise_separation_block(
    inputs,
    output_channels,
    depthwise_kernel_size=3,
    pointwise_kernel_size=1,
    strides=1,
    se_ratio=0.0,
    activation="swish",
    se_activation="relu",
    se_gate_activation="sigmoid",
    se_make_divisible_number=None,
    bn_epsilon=1e-5,
    padding=None,
    name="depthwise_separation_block",
):
    input_channels = inputs.shape[-1]
    has_skip = strides == 1 and input_channels == output_channels

    x = inputs
    x = apply_conv2d_block(
        x,
        kernel_size=depthwise_kernel_size,
        strides=strides,
        activation=activation,
        use_depthwise=True,
        bn_epsilon=bn_epsilon,
        padding=padding,
        name=f"{name}_conv_dw",
    )
    if se_ratio > 0:
        x = apply_se_block(
            x,
            se_ratio,
            activation=se_activation,
            gate_activation=se_gate_activation,
            make_divisible_number=se_make_divisible_number,
            name=f"{name}_se",
        )
    x = apply_conv2d_block(
        x,
        output_channels,
        pointwise_kernel_size,
        1,
        activation=None,
        bn_epsilon=bn_epsilon,
        padding=padding,
        name=f"{name}_conv_pw",
    )
    if has_skip:
        x = layers.Add()([x, inputs])
    return x

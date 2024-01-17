from keras import layers

from kimm.blocks.base_block import apply_conv2d_block
from kimm.blocks.base_block import apply_se_block
from kimm.utils import make_divisible


def apply_inverted_residual_block(
    inputs,
    output_channels,
    depthwise_kernel_size=3,
    expansion_kernel_size=1,
    pointwise_kernel_size=1,
    strides=1,
    expansion_ratio=1.0,
    se_ratio=0.0,
    activation="swish",
    se_channels=None,
    se_activation=None,
    se_gate_activation="sigmoid",
    se_make_divisible_number=None,
    bn_epsilon=1e-5,
    padding=None,
    name="inverted_residual_block",
):
    input_channels = inputs.shape[-1]
    hidden_channels = make_divisible(input_channels * expansion_ratio)
    has_skip = strides == 1 and input_channels == output_channels

    x = inputs
    # Point-wise expansion
    x = apply_conv2d_block(
        x,
        hidden_channels,
        expansion_kernel_size,
        1,
        activation=activation,
        bn_epsilon=bn_epsilon,
        padding=padding,
        name=f"{name}_conv_pw",
    )
    # Depth-wise convolution
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
    # Squeeze-and-excitation
    if se_ratio > 0:
        x = apply_se_block(
            x,
            se_ratio,
            activation=se_activation or activation,
            gate_activation=se_gate_activation,
            se_input_channels=se_channels,
            make_divisible_number=se_make_divisible_number,
            name=f"{name}_se",
        )
    # Point-wise linear projection
    x = apply_conv2d_block(
        x,
        output_channels,
        pointwise_kernel_size,
        1,
        activation=None,
        bn_epsilon=bn_epsilon,
        padding=padding,
        name=f"{name}_conv_pwl",
    )
    if has_skip:
        x = layers.Add()([x, inputs])
    return x

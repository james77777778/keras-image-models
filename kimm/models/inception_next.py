import itertools
import typing

import keras
from keras import backend
from keras import initializers
from keras import layers
from keras import ops

from kimm import layers as kimm_layers
from kimm.blocks import apply_mlp_block
from kimm.models import BaseModel
from kimm.utils import add_model_to_registry


def apply_inception_depthwise_conv2d(
    inputs,
    square_kernel_size: int = 3,
    band_kernel_size: int = 11,
    branch_ratio: float = 0.125,
    name="inception_depthwise_conv2d",
):
    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3
    input_channels = inputs.shape[channels_axis]
    branch_channels = int(input_channels * branch_ratio)
    split_sizes = (
        input_channels - 3 * branch_channels,
        branch_channels,
        branch_channels,
        branch_channels,
    )
    split_indices = list(itertools.accumulate(split_sizes[:-1]))
    square_padding = (square_kernel_size - 1) // 2
    band_padding = (band_kernel_size - 1) // 2

    x = inputs

    x_id, x_hw, x_w, x_h = ops.split(x, split_indices, axis=channels_axis)
    x_hw = layers.ZeroPadding2D(square_padding)(x_hw)
    x_hw = layers.DepthwiseConv2D(
        square_kernel_size,
        use_bias=True,
        name=f"{name}_dwconv_hw_dwconv2d",
    )(x_hw)

    x_w = layers.ZeroPadding2D((0, band_padding))(x_w)
    x_w = layers.DepthwiseConv2D(
        (1, band_kernel_size),
        use_bias=True,
        name=f"{name}_dwconv_w_dwconv2d",
    )(x_w)

    x_h = layers.ZeroPadding2D((band_padding, 0))(x_h)
    x_h = layers.DepthwiseConv2D(
        (band_kernel_size, 1),
        use_bias=True,
        name=f"{name}_dwconv_h_dwconv2d",
    )(x_h)

    x = layers.Concatenate(axis=channels_axis)([x_id, x_hw, x_w, x_h])
    return x


def apply_metanext_block(
    inputs,
    output_channels: int,
    mlp_ratio: float = 4.0,
    activation="gelu",
    name="metanext_block",
):
    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3
    hidden_channels = int(mlp_ratio * output_channels)

    x = inputs

    x = apply_inception_depthwise_conv2d(x, name=f"{name}_token_mixer")
    x = layers.BatchNormalization(
        axis=channels_axis, epsilon=1e-5, name=f"{name}_norm"
    )(x)
    x = apply_mlp_block(
        x,
        hidden_channels,
        output_channels,
        activation,
        use_bias=True,
        use_conv_mlp=True,
        name=f"{name}_mlp",
    )
    x = kimm_layers.LayerScale(
        axis=channels_axis,
        initializer=initializers.Constant(1e-6),
        name=f"{name}_layerscale",
    )(x)

    x = layers.Add()([x, inputs])
    return x


def apply_metanext_stage(
    inputs,
    depth: int,
    output_channels: int,
    strides: int,
    mlp_ratio: float = 4,
    activation="gelu",
    name="convnext_stage",
):
    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3

    x = inputs

    # Downsample
    if strides > 1:
        x = layers.BatchNormalization(
            axis=channels_axis,
            momentum=0.9,
            epsilon=1e-5,
            name=f"{name}_downsample_0",
        )(x)
        x = layers.Conv2D(
            output_channels,
            2,
            strides,
            use_bias=True,
            name=f"{name}_downsample_1_conv2d",
        )(x)

    # Blocks
    for i in range(depth):
        x = apply_metanext_block(
            x,
            output_channels,
            mlp_ratio=mlp_ratio,
            activation=activation,
            name=f"{name}_blocks_{i}",
        )
    return x


@keras.saving.register_keras_serializable(package="kimm")
class InceptionNeXt(BaseModel):
    available_feature_keys = [
        "STEM_S4",
        *[f"BLOCK{i}_S{j}" for i, j in zip(range(4), [4, 8, 16, 32])],
    ]

    def __init__(
        self,
        depths: typing.Sequence[int] = [3, 3, 9, 3],
        hidden_channels: typing.Sequence[int] = [96, 192, 384, 768],
        mlp_ratios: typing.Sequence[float] = [4, 4, 4, 3],
        activation: str = "gelu",
        **kwargs,
    ):
        kwargs["weights_url"] = self.get_weights_url(kwargs["weights"])

        input_tensor = kwargs.pop("input_tensor", None)
        self.set_properties(kwargs, 224)
        channels_axis = (
            -1 if backend.image_data_format() == "channels_last" else -3
        )

        inputs = self.determine_input_tensor(
            input_tensor,
            self._input_shape,
            self._default_size,
            require_flatten=self._include_top,
        )
        x = inputs

        x = self.build_preprocessing(x, "imagenet")

        # Prepare feature extraction
        features = {}

        # Stem
        x = layers.Conv2D(
            hidden_channels[0], 4, 4, use_bias=True, name="stem_0_conv2d"
        )(x)
        x = layers.BatchNormalization(
            axis=channels_axis, momentum=0.9, epsilon=1e-5, name="stem_1"
        )(x)
        features["STEM_S4"] = x

        # Blocks (4 stages)
        current_stride = 4
        for i in range(4):
            strides = 2 if i > 0 else 1
            x = apply_metanext_stage(
                x,
                depths[i],
                hidden_channels[i],
                strides,
                mlp_ratios[i],
                activation=activation,
                name=f"stages_{i}",
            )
            current_stride *= strides
            # Add feature
            features[f"BLOCK{i}_S{current_stride}"] = x

        # Head
        x = self.build_head(x)
        super().__init__(inputs=inputs, outputs=x, features=features, **kwargs)

        # All references to `self` below this line
        self.depths = depths
        self.hidden_channels = hidden_channels
        self.mlp_ratios = mlp_ratios
        self.activation = activation

    def build_top(self, inputs, classes, classifier_activation, dropout_rate):
        channels_axis = (
            -1 if backend.image_data_format() == "channels_last" else -3
        )
        input_channels = inputs.shape[channels_axis]
        hidden_channels = int(input_channels * 3.0)

        x = inputs
        x = layers.GlobalAveragePooling2D(name="avg_pool")(inputs)
        x = layers.Dense(hidden_channels, use_bias=True, name="head_fc1")(x)
        x = layers.Activation("gelu")(x)
        x = layers.LayerNormalization(axis=-1, epsilon=1e-6, name="head_norm")(
            x
        )
        x = layers.Dropout(rate=dropout_rate, name="head_dropout")(x)
        x = layers.Dense(
            classes, activation=classifier_activation, name="classifier"
        )(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "depths": self.depths,
                "hidden_channels": self.hidden_channels,
                "mlp_ratios": self.mlp_ratios,
                "activation": self.activation,
            }
        )
        return config

    def fix_config(self, config: typing.Dict):
        unused_kwargs = [
            "depths",
            "hidden_channels",
            "mlp_ratios",
            "activation",
        ]
        for k in unused_kwargs:
            config.pop(k, None)
        return config


"""
Model Definition
"""


class InceptionNeXtTiny(InceptionNeXt):
    available_weights = [
        (
            "imagenet",
            InceptionNeXt.default_origin,
            "inceptionnexttiny_inception_next_tiny.sail_in1k.keras",
        )
    ]

    def __init__(
        self,
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        dropout_rate: float = 0.0,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = "imagenet",
        name: str = "InceptionNeXtTiny",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            (3, 3, 9, 3),
            (96, 192, 384, 768),
            (4, 4, 4, 3),
            "gelu",
            input_tensor=input_tensor,
            input_shape=input_shape,
            include_preprocessing=include_preprocessing,
            include_top=include_top,
            pooling=pooling,
            dropout_rate=dropout_rate,
            classes=classes,
            classifier_activation=classifier_activation,
            weights=weights,
            name=name,
            **kwargs,
        )


class InceptionNeXtSmall(InceptionNeXt):
    available_weights = [
        (
            "imagenet",
            InceptionNeXt.default_origin,
            "inceptionnextsmall_inception_next_small.sail_in1k.keras",
        )
    ]

    def __init__(
        self,
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        dropout_rate: float = 0.0,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = "imagenet",
        name: str = "InceptionNeXtSmall",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            (3, 3, 27, 3),
            (96, 192, 384, 768),
            (4, 4, 4, 3),
            "gelu",
            input_tensor=input_tensor,
            input_shape=input_shape,
            include_preprocessing=include_preprocessing,
            include_top=include_top,
            pooling=pooling,
            dropout_rate=dropout_rate,
            classes=classes,
            classifier_activation=classifier_activation,
            weights=weights,
            name=name,
            **kwargs,
        )


class InceptionNeXtBase(InceptionNeXt):
    available_weights = [
        (
            "imagenet",
            InceptionNeXt.default_origin,
            "inceptionnextbase_inception_next_base.sail_in1k_384.keras",
        )
    ]

    def __init__(
        self,
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        dropout_rate: float = 0.0,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = "imagenet",
        name: str = "InceptionNeXtBase",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            (3, 3, 27, 3),
            (128, 256, 512, 1024),
            (4, 4, 4, 3),
            "gelu",
            input_tensor=input_tensor,
            input_shape=input_shape,
            include_preprocessing=include_preprocessing,
            include_top=include_top,
            pooling=pooling,
            dropout_rate=dropout_rate,
            classes=classes,
            classifier_activation=classifier_activation,
            weights=weights,
            name=name,
            default_size=384,
            **kwargs,
        )


add_model_to_registry(InceptionNeXtTiny, "imagenet")
add_model_to_registry(InceptionNeXtSmall, "imagenet")
add_model_to_registry(InceptionNeXtBase, "imagenet")

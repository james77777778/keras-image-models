import typing

import keras
from keras import backend
from keras import layers

from kimm.models.base_model import BaseModel
from kimm.utils import add_model_to_registry


def apply_convmixer_block(
    inputs, output_channels, kernel_size, activation, name="convmixer_block"
):
    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3

    x = inputs

    # Depthwise
    x = layers.DepthwiseConv2D(
        kernel_size,
        1,
        padding="same",
        activation=activation,
        use_bias=True,
        name=f"{name}_0_fn_0_dwconv2d",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis, momentum=0.9, epsilon=1e-5, name=f"{name}_0_fn_2"
    )(x)
    x = layers.Add()([x, inputs])

    # Pointwise
    x = layers.Conv2D(
        output_channels,
        1,
        1,
        activation=activation,
        use_bias=True,
        name=f"{name}_1_conv2d",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis, momentum=0.9, epsilon=1e-5, name=f"{name}_3"
    )(x)
    return x


@keras.saving.register_keras_serializable(package="kimm")
class ConvMixer(BaseModel):
    def __init__(
        self,
        depth: int = 32,
        hidden_channels: int = 768,
        patch_size: int = 7,
        kernel_size: int = 7,
        activation: str = "relu",
        **kwargs,
    ):
        kwargs["weights_url"] = self.get_weights_url(kwargs["weights"])
        input_tensor = kwargs.pop("input_tensor", None)
        self.set_properties(kwargs)
        channels_axis = (
            -1 if backend.image_data_format() == "channels_last" else -3
        )

        inputs = self.determine_input_tensor(
            input_tensor,
            self._input_shape,
            self._default_size,
        )
        x = inputs

        x = self.build_preprocessing(x, "imagenet")

        # Prepare feature extraction
        features = {}

        # Stem
        x = layers.Conv2D(
            hidden_channels,
            patch_size,
            patch_size,
            activation=activation,
            use_bias=True,
            name="stem_conv2d",
        )(x)
        x = layers.BatchNormalization(
            axis=channels_axis, momentum=0.9, epsilon=1e-5, name="stem_bn"
        )(x)
        features["STEM"] = x

        # Blocks
        for i in range(depth):
            x = apply_convmixer_block(
                x, hidden_channels, kernel_size, activation, name=f"blocks_{i}"
            )
            # Add feature
            features[f"BLOCK{i}"] = x

        # Head
        x = self.build_head(x)

        super().__init__(inputs=inputs, outputs=x, features=features, **kwargs)

        # All references to `self` below this line
        self.depth = depth
        self.hidden_channels = hidden_channels
        self.patch_size = patch_size
        self.kernel_size = kernel_size
        self.activation = activation

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "depth": self.depth,
                "hidden_channels": self.hidden_channels,
                "patch_size": self.patch_size,
                "kernel_size": self.kernel_size,
                "activation": self.activation,
            }
        )
        return config

    def fix_config(self, config):
        unused_kwargs = [
            "depth",
            "hidden_channels",
            "patch_size",
            "kernel_size",
            "activation",
        ]
        for k in unused_kwargs:
            config.pop(k, None)
        return config


"""
Model Definition
"""


class ConvMixer736D32(ConvMixer):
    available_feature_keys = ["STEM", *[f"BLOCK{i}" for i in range(32)]]
    available_weights = [
        (
            "imagenet",
            ConvMixer.default_origin,
            "convmixer736d32_convmixer_768_32.in1k.keras",
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
        name: str = "ConvMixer736D32",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            32,
            768,
            7,
            7,
            "relu",
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


class ConvMixer1024D20(ConvMixer):
    available_feature_keys = ["STEM", *[f"BLOCK{i}" for i in range(20)]]
    available_weights = [
        (
            "imagenet",
            ConvMixer.default_origin,
            "convmixer1024d20_convmixer_1024_20_ks9_p14.in1k.keras",
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
        name: str = "ConvMixer1024D20",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            20,
            1024,
            14,
            9,
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


class ConvMixer1536D20(ConvMixer):
    available_feature_keys = ["STEM", *[f"BLOCK{i}" for i in range(20)]]
    available_weights = [
        (
            "imagenet",
            ConvMixer.default_origin,
            "convmixer1536d20_convmixer_1536_20.in1k.keras",
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
        name: str = "ConvMixer1536D20",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            20,
            1536,
            7,
            9,
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


add_model_to_registry(ConvMixer736D32, "imagenet")
add_model_to_registry(ConvMixer1024D20, "imagenet")
add_model_to_registry(ConvMixer1536D20, "imagenet")

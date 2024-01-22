import typing

import keras
from keras import backend
from keras import layers

from kimm.models import BaseModel
from kimm.utils import add_model_to_registry

DEFAULT_VGG11_CONFIG = [
    64,
    "M",
    128,
    "M",
    256,
    256,
    "M",
    512,
    512,
    "M",
    512,
    512,
    "M",
]
DEFAULT_VGG13_CONFIG = [
    64,
    64,
    "M",
    128,
    128,
    "M",
    256,
    256,
    "M",
    512,
    512,
    "M",
    512,
    512,
    "M",
]
DEFAULT_VGG16_CONFIG = [
    64,
    64,
    "M",
    128,
    128,
    "M",
    256,
    256,
    256,
    "M",
    512,
    512,
    512,
    "M",
    512,
    512,
    512,
    "M",
]
DEFAULT_VGG19_CONFIG = [
    64,
    64,
    "M",
    128,
    128,
    "M",
    256,
    256,
    256,
    256,
    "M",
    512,
    512,
    512,
    512,
    "M",
    512,
    512,
    512,
    512,
    "M",
]


def apply_conv_mlp_layer(
    inputs,
    output_channels,
    kernel_size,
    mlp_ratio=1.0,
    dropout_rate=0.2,
    name="conv_mlp_layer",
):
    mid_channels = int(output_channels * mlp_ratio)

    x = inputs
    x = layers.Conv2D(
        mid_channels, kernel_size, 1, use_bias=True, name=f"{name}_fc1conv2d"
    )(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rate, name=f"{name}_drop")(x)
    x = layers.Conv2D(
        output_channels, 1, 1, use_bias=True, name=f"{name}_fc2conv2d"
    )(x)
    x = layers.ReLU()(x)
    return x


@keras.saving.register_keras_serializable(package="kimm")
class VGG(BaseModel):
    available_feature_keys = [
        *[f"BLOCK{i}_S{j}" for i, j in zip(range(6), [1, 2, 4, 8, 16, 32])],
    ]

    def __init__(self, config: typing.Union[str, typing.List], **kwargs):
        kwargs["weights_url"] = self.get_weights_url(kwargs["weights"])

        _available_configs = ["vgg11", "vgg13", "vgg16", "vgg19"]
        if config == "vgg11":
            _config = DEFAULT_VGG11_CONFIG
        elif config == "vgg13":
            _config = DEFAULT_VGG13_CONFIG
        elif config == "vgg16":
            _config = DEFAULT_VGG16_CONFIG
        elif config == "vgg19":
            _config = DEFAULT_VGG19_CONFIG
        else:
            raise ValueError(
                f"config must be one of {_available_configs} using string. "
                f"Received: config={config}"
            )

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

        # Blocks
        current_stage_idx = 0
        current_block_idx = 0
        current_stride = 1
        for c in _config:
            name = f"features_{current_block_idx}"
            if c == "M":
                features[f"BLOCK{current_stage_idx}_S{current_stride}"] = x
                x = layers.MaxPooling2D(2, 2, name=name)(x)
                current_stride *= 2
                current_stage_idx += 1
                current_block_idx += 1
            else:
                x = layers.Conv2D(
                    c,
                    3,
                    1,
                    padding="same",
                    use_bias=True,
                    name=f"features_{current_block_idx}conv2d",
                )(x)
                x = layers.BatchNormalization(
                    axis=channels_axis,
                    momentum=0.9,
                    epsilon=1e-5,
                    name=f"features_{current_block_idx + 1}",
                )(x)
                x = layers.ReLU(name=f"features_{current_block_idx + 2}")(x)
                current_block_idx += 3

        features[f"BLOCK{current_stage_idx}_S{current_stride}"] = x
        x = apply_conv_mlp_layer(x, 4096, 7, 1.0, 0.0, name="pre_logits")

        # Head
        x = self.build_head(x)

        super().__init__(inputs=inputs, outputs=x, features=features, **kwargs)

        # All references to `self` below this line
        self.config = config

    def get_config(self):
        config = super().get_config()
        config.update({"config": self.config})
        return config

    def fix_config(self, config: typing.Dict):
        unused_kwargs = ["config"]
        for k in unused_kwargs:
            config.pop(k, None)
        return config


"""
Model Definition
"""


class VGG11(VGG):
    available_weights = [
        (
            "imagenet",
            VGG.default_origin,
            "vgg11_vgg11_bn.tv_in1k.keras",
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
        name: str = "VGG11",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            "vgg11",
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


class VGG13(VGG):
    available_weights = [
        (
            "imagenet",
            VGG.default_origin,
            "vgg13_vgg13_bn.tv_in1k.keras",
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
        name: str = "VGG13",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            "vgg13",
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


class VGG16(VGG):
    available_weights = [
        (
            "imagenet",
            VGG.default_origin,
            "vgg16_vgg16_bn.tv_in1k.keras",
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
        name: str = "VGG16",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            "vgg16",
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


class VGG19(VGG):
    available_weights = [
        (
            "imagenet",
            VGG.default_origin,
            "vgg19_vgg19_bn.tv_in1k.keras",
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
        name: str = "VGG19",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            "vgg19",
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


add_model_to_registry(VGG11, "imagenet")
add_model_to_registry(VGG13, "imagenet")
add_model_to_registry(VGG16, "imagenet")
add_model_to_registry(VGG19, "imagenet")

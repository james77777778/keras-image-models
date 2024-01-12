import math
import typing

import keras
from keras import backend
from keras import layers
from keras import utils
from keras.src.applications import imagenet_utils

from kimm.blocks import apply_conv2d_block
from kimm.blocks import apply_se_block
from kimm.models.feature_extractor import FeatureExtractor
from kimm.utils import make_divisible

DEFAULT_CONFIG = [
    # type, repeat, kernel_size, strides, expansion_ratio, channels, se_ratio
    ["ds", 1, 3, 1, 1, 16, 0.25],
    ["ir", 2, 3, 2, 6, 24, 0.25],
    ["ir", 2, 5, 2, 6, 40, 0.25],
    ["ir", 3, 3, 2, 6, 80, 0.25],
    ["ir", 3, 5, 1, 6, 112, 0.25],
    ["ir", 4, 5, 2, 6, 192, 0.25],
    ["ir", 1, 3, 1, 6, 320, 0.25],
]
DEFAULT_LITE_CONFIG = [
    # type, repeat, kernel_size, strides, expansion_ratio, channels, se_ratio
    ["ds", 1, 3, 1, 1, 16, 0.0],
    ["ir", 2, 3, 2, 6, 24, 0.0],
    ["ir", 2, 5, 2, 6, 40, 0.0],
    ["ir", 3, 3, 2, 6, 80, 0.0],
    ["ir", 3, 5, 1, 6, 112, 0.0],
    ["ir", 4, 5, 2, 6, 192, 0.0],
    ["ir", 1, 3, 1, 6, 320, 0.0],
]


def apply_depthwise_separation_block(
    inputs,
    output_channels,
    depthwise_kernel_size=3,
    pointwise_kernel_size=1,
    strides=1,
    se_ratio=0.0,
    activation="swish",
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
            activation=activation,
            gate_activation="sigmoid",
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
            activation=activation,
            gate_activation="sigmoid",
            se_input_channels=input_channels,
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


class EfficientNet(FeatureExtractor):
    def __init__(
        self,
        width: float = 1.0,
        depth: float = 1.0,
        fix_stem_and_head_channels: bool = False,
        fix_first_and_last_blocks: bool = False,
        activation="swish",
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        dropout_rate: float = 0.0,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = None,  # TODO: imagenet
        config: typing.Union[str, typing.List] = "default",
        **kwargs,
    ):
        if config == "default":
            config = DEFAULT_CONFIG
        elif config == "lite":
            config = DEFAULT_LITE_CONFIG
        else:
            if isinstance(config, str):
                raise ValueError(
                    "config must be one of ('default', 'lite') using string. "
                    f"Received: config={config}"
                )
        # TF default config
        default_size = kwargs.pop("default_size", 224)
        bn_epsilon = kwargs.pop("bn_epsilon", 1e-5)
        padding = kwargs.pop("padding", None)

        # Prepare feature extraction
        features = {}

        # Determine proper input shape
        input_shape = imagenet_utils.obtain_input_shape(
            input_shape,
            default_size=default_size,
            min_size=32,
            data_format=backend.image_data_format(),
            require_flatten=include_top,
            weights=weights,
        )

        if input_tensor is None:
            img_input = layers.Input(shape=input_shape)
        else:
            if not backend.is_keras_tensor(input_tensor):
                img_input = layers.Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        x = img_input

        # [0, 255] to [0, 1] and apply ImageNet mean and variance
        if include_preprocessing:
            x = layers.Rescaling(scale=1.0 / 255.0)(x)
            x = layers.Normalization(
                mean=[0.485, 0.456, 0.406], variance=[0.229, 0.224, 0.225]
            )(x)

        # stem
        stem_channel = (
            32 if fix_stem_and_head_channels else make_divisible(32 * width)
        )
        x = apply_conv2d_block(
            x,
            stem_channel,
            3,
            2,
            activation=activation,
            bn_epsilon=bn_epsilon,
            padding=padding,
            name="conv_stem",
        )
        features["STEM_S2"] = x

        # blocks
        current_stride = 2
        for current_block_idx, cfg in enumerate(config):
            block_type, r, k, s, e, c, se = cfg
            c = make_divisible(c * width)
            if fix_first_and_last_blocks and (
                current_block_idx in (0, len(config) - 1)
            ):
                r = r
            else:
                r = int(math.ceil(r * depth))
            for current_layer_idx in range(r):
                s = s if current_layer_idx == 0 else 1
                common_kwargs = {
                    "bn_epsilon": bn_epsilon,
                    "padding": padding,
                    "name": f"blocks_{current_block_idx}_{current_layer_idx}",
                }
                if block_type == "ds":
                    x = apply_depthwise_separation_block(
                        x, c, k, 1, s, se, activation, **common_kwargs
                    )
                elif block_type == "ir":
                    x = apply_inverted_residual_block(
                        x, c, k, 1, 1, s, e, se, activation, **common_kwargs
                    )
                elif block_type == "cn":
                    x = apply_conv2d_block(
                        x,
                        filters=c,
                        kernel_size=k,
                        strides=s,
                        activation=activation,
                        **common_kwargs,
                    )
                current_stride *= s
            features[f"BLOCK{current_block_idx}_S{current_stride}"] = x

        # last conv
        if fix_stem_and_head_channels:
            conv_head_channels = 1280
        else:
            conv_head_channels = make_divisible(1280 * width)
        x = apply_conv2d_block(
            x,
            conv_head_channels,
            1,
            1,
            activation=activation,
            bn_epsilon=bn_epsilon,
            name="conv_head",
        )

        if include_top:
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            x = layers.Dropout(rate=dropout_rate, name="conv_head_dropout")(x)
            x = layers.Dense(
                classes, activation=classifier_activation, name="classifier"
            )(x)
        else:
            if pooling == "avg":
                x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            elif pooling == "max":
                x = layers.GlobalMaxPooling2D(name="max_pool")(x)

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        if input_tensor is not None:
            inputs = utils.get_source_inputs(input_tensor)
        else:
            inputs = img_input

        super().__init__(inputs=inputs, outputs=x, features=features, **kwargs)

        # All references to `self` below this line
        self.width = width
        self.depth = depth
        self.fix_stem_and_head_channels = fix_stem_and_head_channels
        self.include_preprocessing = include_preprocessing
        self.include_top = include_top
        self.pooling = pooling
        self.dropout_rate = dropout_rate
        self.classes = classes
        self.classifier_activation = classifier_activation
        self._weights = weights  # `self.weights` is been used internally
        self.config = config

    @staticmethod
    def available_feature_keys():
        feature_keys = ["STEM_S2"]
        feature_keys.extend(
            [
                f"BLOCK{i}_S{j}"
                for i, j in zip(range(7), [2, 4, 8, 16, 16, 32, 32])
            ]
        )
        return feature_keys

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "width": self.width,
                "input_shape": self.input_shape[1:],
                "include_preprocessing": self.include_preprocessing,
                "include_top": self.include_top,
                "pooling": self.pooling,
                "dropout_rate": self.dropout_rate,
                "classes": self.classes,
                "classifier_activation": self.classifier_activation,
                "weights": self._weights,
                "config": self.config,
            }
        )
        return config


"""
Model Definition
"""


class EfficientNetB0(EfficientNet):
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
        weights: typing.Optional[str] = None,  # TODO: imagenet
        config: typing.Union[str, typing.List] = "default",
        name: str = "EfficientNetB0",
        **kwargs,
    ):
        # default to TF configuration (bn_epsilon=1e-3 and padding="same")
        super().__init__(
            1.0,
            1.0,
            False,
            False,
            "swish",
            input_tensor,
            input_shape,
            include_preprocessing,
            include_top,
            pooling,
            dropout_rate,
            classes,
            classifier_activation,
            weights,
            config,
            name=name,
            default_size=224,
            bn_epsilon=1e-3,
            padding="same",
            **kwargs,
        )


class EfficientNetB1(EfficientNet):
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
        weights: typing.Optional[str] = None,  # TODO: imagenet
        config: typing.Union[str, typing.List] = "default",
        name: str = "EfficientNetB1",
        **kwargs,
    ):
        # default to TF configuration (bn_epsilon=1e-3 and padding="same")
        super().__init__(
            1.0,
            1.1,
            False,
            False,
            "swish",
            input_tensor,
            input_shape,
            include_preprocessing,
            include_top,
            pooling,
            dropout_rate,
            classes,
            classifier_activation,
            weights,
            config,
            name=name,
            default_size=240,
            bn_epsilon=1e-3,
            padding="same",
            **kwargs,
        )


class EfficientNetB2(EfficientNet):
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
        weights: typing.Optional[str] = None,  # TODO: imagenet
        config: typing.Union[str, typing.List] = "default",
        name: str = "EfficientNetB2",
        **kwargs,
    ):
        # default to TF configuration (bn_epsilon=1e-3 and padding="same")
        super().__init__(
            1.1,
            1.2,
            False,
            False,
            "swish",
            input_tensor,
            input_shape,
            include_preprocessing,
            include_top,
            pooling,
            dropout_rate,
            classes,
            classifier_activation,
            weights,
            config,
            name=name,
            default_size=260,
            bn_epsilon=1e-3,
            padding="same",
            **kwargs,
        )


class EfficientNetB3(EfficientNet):
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
        weights: typing.Optional[str] = None,  # TODO: imagenet
        config: typing.Union[str, typing.List] = "default",
        name: str = "EfficientNetB3",
        **kwargs,
    ):
        # default to TF configuration (bn_epsilon=1e-3 and padding="same")
        super().__init__(
            1.2,
            1.4,
            False,
            False,
            "swish",
            input_tensor,
            input_shape,
            include_preprocessing,
            include_top,
            pooling,
            dropout_rate,
            classes,
            classifier_activation,
            weights,
            config,
            name=name,
            default_size=300,
            bn_epsilon=1e-3,
            padding="same",
            **kwargs,
        )


class EfficientNetB4(EfficientNet):
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
        weights: typing.Optional[str] = None,  # TODO: imagenet
        config: typing.Union[str, typing.List] = "default",
        name: str = "EfficientNetB4",
        **kwargs,
    ):
        # default to TF configuration (bn_epsilon=1e-3 and padding="same")
        super().__init__(
            1.4,
            1.8,
            False,
            False,
            "swish",
            input_tensor,
            input_shape,
            include_preprocessing,
            include_top,
            pooling,
            dropout_rate,
            classes,
            classifier_activation,
            weights,
            config,
            name=name,
            default_size=380,
            bn_epsilon=1e-3,
            padding="same",
            **kwargs,
        )


class EfficientNetB5(EfficientNet):
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
        weights: typing.Optional[str] = None,  # TODO: imagenet
        config: typing.Union[str, typing.List] = "default",
        name: str = "EfficientNetB5",
        **kwargs,
    ):
        # default to TF configuration (bn_epsilon=1e-3 and padding="same")
        super().__init__(
            1.6,
            2.2,
            False,
            False,
            "swish",
            input_tensor,
            input_shape,
            include_preprocessing,
            include_top,
            pooling,
            dropout_rate,
            classes,
            classifier_activation,
            weights,
            config,
            name=name,
            default_size=456,
            bn_epsilon=1e-3,
            padding="same",
            **kwargs,
        )


class EfficientNetB6(EfficientNet):
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
        weights: typing.Optional[str] = None,  # TODO: imagenet
        config: typing.Union[str, typing.List] = "default",
        name: str = "EfficientNetB6",
        **kwargs,
    ):
        # default to TF configuration (bn_epsilon=1e-3 and padding="same")
        super().__init__(
            1.8,
            2.6,
            False,
            False,
            "swish",
            input_tensor,
            input_shape,
            include_preprocessing,
            include_top,
            pooling,
            dropout_rate,
            classes,
            classifier_activation,
            weights,
            config,
            name=name,
            default_size=528,
            bn_epsilon=1e-3,
            padding="same",
            **kwargs,
        )


class EfficientNetB7(EfficientNet):
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
        weights: typing.Optional[str] = None,  # TODO: imagenet
        config: typing.Union[str, typing.List] = "default",
        name: str = "EfficientNetB7",
        **kwargs,
    ):
        # default to TF configuration (bn_epsilon=1e-3 and padding="same")
        super().__init__(
            2.0,
            3.1,
            False,
            False,
            "swish",
            input_tensor,
            input_shape,
            include_preprocessing,
            include_top,
            pooling,
            dropout_rate,
            classes,
            classifier_activation,
            weights,
            config,
            name=name,
            default_size=600,
            bn_epsilon=1e-3,
            padding="same",
            **kwargs,
        )


class EfficientNetLiteB0(EfficientNet):
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
        weights: typing.Optional[str] = None,  # TODO: imagenet
        config: typing.Union[str, typing.List] = "lite",
        name: str = "EfficientNetLiteB0",
        **kwargs,
    ):
        # default to TF configuration (bn_epsilon=1e-3 and padding="same")
        super().__init__(
            1.0,
            1.0,
            True,
            True,
            "relu6",
            input_tensor,
            input_shape,
            include_preprocessing,
            include_top,
            pooling,
            dropout_rate,
            classes,
            classifier_activation,
            weights,
            config,
            name=name,
            default_size=224,
            bn_epsilon=1e-3,
            padding="same",
            **kwargs,
        )


class EfficientNetLiteB1(EfficientNet):
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
        weights: typing.Optional[str] = None,  # TODO: imagenet
        config: typing.Union[str, typing.List] = "lite",
        name: str = "EfficientNetLiteB1",
        **kwargs,
    ):
        # default to TF configuration (bn_epsilon=1e-3 and padding="same")
        super().__init__(
            1.0,
            1.1,
            True,
            True,
            "relu6",
            input_tensor,
            input_shape,
            include_preprocessing,
            include_top,
            pooling,
            dropout_rate,
            classes,
            classifier_activation,
            weights,
            config,
            name=name,
            default_size=240,
            bn_epsilon=1e-3,
            padding="same",
            **kwargs,
        )


class EfficientNetLiteB2(EfficientNet):
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
        weights: typing.Optional[str] = None,  # TODO: imagenet
        config: typing.Union[str, typing.List] = "lite",
        name: str = "EfficientNetLiteB2",
        **kwargs,
    ):
        # default to TF configuration (bn_epsilon=1e-3 and padding="same")
        super().__init__(
            1.1,
            1.2,
            True,
            True,
            "relu6",
            input_tensor,
            input_shape,
            include_preprocessing,
            include_top,
            pooling,
            dropout_rate,
            classes,
            classifier_activation,
            weights,
            config,
            name=name,
            default_size=260,
            bn_epsilon=1e-3,
            padding="same",
            **kwargs,
        )


class EfficientNetLiteB3(EfficientNet):
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
        weights: typing.Optional[str] = None,  # TODO: imagenet
        config: typing.Union[str, typing.List] = "lite",
        name: str = "EfficientNetLiteB3",
        **kwargs,
    ):
        # default to TF configuration (bn_epsilon=1e-3 and padding="same")
        super().__init__(
            1.2,
            1.4,
            True,
            True,
            "relu6",
            input_tensor,
            input_shape,
            include_preprocessing,
            include_top,
            pooling,
            dropout_rate,
            classes,
            classifier_activation,
            weights,
            config,
            name=name,
            default_size=300,
            bn_epsilon=1e-3,
            padding="same",
            **kwargs,
        )


class EfficientNetLiteB4(EfficientNet):
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
        weights: typing.Optional[str] = None,  # TODO: imagenet
        config: typing.Union[str, typing.List] = "lite",
        name: str = "EfficientNetLiteB4",
        **kwargs,
    ):
        # default to TF configuration (bn_epsilon=1e-3 and padding="same")
        super().__init__(
            1.4,
            1.8,
            True,
            True,
            "relu6",
            input_tensor,
            input_shape,
            include_preprocessing,
            include_top,
            pooling,
            dropout_rate,
            classes,
            classifier_activation,
            weights,
            config,
            name=name,
            default_size=380,
            bn_epsilon=1e-3,
            padding="same",
            **kwargs,
        )

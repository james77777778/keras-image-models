import math
import typing

import keras
from keras import backend
from keras import layers
from keras import utils
from keras.src.applications import imagenet_utils

from kimm.blocks import apply_conv2d_block
from kimm.blocks import apply_depthwise_separation_block
from kimm.blocks import apply_inverted_residual_block
from kimm.models.feature_extractor import FeatureExtractor
from kimm.utils import add_model_to_registry
from kimm.utils import make_divisible

DEFAULT_CONFIG = [
    # type, repeat, kernel_size, strides, expansion_ratio, channels
    ["ds", 1, 3, 1, 1, 16],
    ["ir", 2, 3, 2, 6, 24],
    ["ir", 3, 3, 2, 6, 32],
    ["ir", 4, 3, 2, 6, 64],
    ["ir", 3, 3, 1, 6, 96],
    ["ir", 3, 3, 2, 6, 160],
    ["ir", 1, 3, 1, 6, 320],
]


class MobileNetV2(FeatureExtractor):
    def __init__(
        self,
        width: float = 1.0,
        depth: float = 1.0,
        fix_stem_and_head_channels: bool = False,
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

        # Prepare feature extraction
        features = {}

        # Determine proper input shape
        input_shape = imagenet_utils.obtain_input_shape(
            input_shape,
            default_size=224,
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
            activation="relu6",
            name="conv_stem",
        )
        features["STEM_S2"] = x

        # blocks
        current_stride = 2
        for current_block_idx, cfg in enumerate(config):
            block_type, r, k, s, e, c = cfg
            c = make_divisible(c * width)
            # no depth multiplier at first and last block
            if current_block_idx not in (0, len(config) - 1):
                r = int(math.ceil(r * depth))
            for current_layer_idx in range(r):
                s = s if current_layer_idx == 0 else 1
                name = f"blocks_{current_block_idx}_{current_layer_idx}"
                if block_type == "ds":
                    x = apply_depthwise_separation_block(
                        x,
                        c,
                        k,
                        1,
                        s,
                        activation="relu6",
                        name=name,
                    )
                elif block_type == "ir":
                    x = apply_inverted_residual_block(
                        x, c, k, 1, 1, s, e, activation="relu6", name=name
                    )
                current_stride *= s
            features[f"BLOCK{current_block_idx}_S{current_stride}"] = x

        # last conv
        if fix_stem_and_head_channels:
            head_channels = 1280
        else:
            head_channels = max(1280, make_divisible(1280 * width))
        x = apply_conv2d_block(
            x, head_channels, 1, 1, activation="relu6", name="conv_head"
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


class MobileNet050V2(MobileNetV2):
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
        name: str = "MobileNet050V2",
        **kwargs,
    ):
        super().__init__(
            0.5,
            1.0,
            False,
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
            **kwargs,
        )


class MobileNet100V2(MobileNetV2):
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
        name: str = "MobileNet100V2",
        **kwargs,
    ):
        super().__init__(
            1.0,
            1.0,
            False,
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
            **kwargs,
        )


class MobileNet110V2(MobileNetV2):
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
        name: str = "MobileNet110V2",
        **kwargs,
    ):
        super().__init__(
            1.1,
            1.2,
            True,
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
            **kwargs,
        )


class MobileNet120V2(MobileNetV2):
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
        name: str = "MobileNet120V2",
        **kwargs,
    ):
        super().__init__(
            1.2,
            1.4,
            True,
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
            **kwargs,
        )


class MobileNet140V2(MobileNetV2):
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
        name: str = "MobileNet140V2",
        **kwargs,
    ):
        super().__init__(
            1.4,
            1.0,
            False,
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
            **kwargs,
        )


add_model_to_registry(MobileNet050V2, True)
add_model_to_registry(MobileNet100V2, True)
add_model_to_registry(MobileNet110V2, True)
add_model_to_registry(MobileNet120V2, True)
add_model_to_registry(MobileNet140V2, True)

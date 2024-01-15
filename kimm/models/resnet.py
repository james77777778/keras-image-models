import typing

import keras
from keras import backend
from keras import layers
from keras import utils
from keras.src.applications import imagenet_utils

from kimm.blocks import apply_conv2d_block
from kimm.models.feature_extractor import FeatureExtractor
from kimm.utils import add_model_to_registry


def apply_basic_block(
    inputs,
    output_channels: int,
    strides: int = 1,
    activation="relu",
    name="basic_block",
):
    input_channels = inputs.shape[-1]
    shortcut = inputs
    x = inputs
    x = apply_conv2d_block(
        x,
        output_channels,
        3,
        strides,
        activation=activation,
        name=f"{name}_conv1",
    )
    x = apply_conv2d_block(
        x,
        output_channels,
        3,
        1,
        activation=None,
        name=f"{name}_conv2",
    )

    # downsampling
    if strides != 1 or input_channels != output_channels:
        shortcut = apply_conv2d_block(
            shortcut,
            output_channels,
            1,
            strides,
            activation=None,
            name=f"{name}_downsample",
        )

    x = layers.Add(name=f"{name}_add")([x, shortcut])
    x = layers.Activation(activation=activation, name=f"{name}")(x)
    return x


def apply_bottleneck_block(
    inputs,
    output_channels: int,
    strides: int = 1,
    activation="relu",
    name="bottleneck_block",
):
    input_channels = inputs.shape[-1]
    expansion = 4
    shortcut = inputs
    x = inputs
    x = apply_conv2d_block(
        x,
        output_channels,
        1,
        1,
        activation=activation,
        name=f"{name}_conv1",
    )
    x = apply_conv2d_block(
        x,
        output_channels,
        3,
        strides,
        activation=activation,
        name=f"{name}_conv2",
    )
    x = apply_conv2d_block(
        x,
        output_channels * expansion,
        1,
        1,
        activation=None,
        name=f"{name}_conv3",
    )

    # downsampling
    if strides != 1 or input_channels != output_channels * expansion:
        shortcut = apply_conv2d_block(
            shortcut,
            output_channels * expansion,
            1,
            strides,
            activation=None,
            name=f"{name}_downsample",
        )

    x = layers.Add(name=f"{name}_add")([x, shortcut])
    x = layers.Activation(activation=activation, name=f"{name}")(x)
    return x


class ResNet(FeatureExtractor):
    def __init__(
        self,
        block_fn: str,
        num_blocks: typing.Sequence[int],
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        dropout_rate: float = 0.0,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = None,  # TODO: imagenet
        **kwargs,
    ):
        if block_fn not in ("basic", "bottleneck"):
            raise ValueError(
                "`block_fn` must be one of ('basic', 'bottelneck'). "
                f"Received: block_fn={block_fn}"
            )
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
        stem_channels = 64
        x = apply_conv2d_block(
            x, stem_channels, 7, 2, activation="relu", name="conv_stem"
        )
        features["STEM_S2"] = x

        # max pooling
        x = layers.ZeroPadding2D(padding=1)(x)
        x = layers.MaxPooling2D(3, strides=2)(x)

        # stages
        output_channels = [64, 128, 256, 512]
        current_stride = 4
        for current_stage_idx, (c, n) in enumerate(
            zip(output_channels, num_blocks)
        ):
            stride = 1 if current_stage_idx == 0 else 2
            current_stride *= stride
            # blocks
            for current_block_idx in range(n):
                stride = stride if current_block_idx == 0 else 1
                name = f"layer{current_stage_idx + 1}_{current_block_idx}"
                if block_fn == "basic":
                    x = apply_basic_block(x, c, stride, name=name)
                elif block_fn == "bottleneck":
                    x = apply_bottleneck_block(x, c, stride, name=name)
                else:
                    raise NotImplementedError
            # add feature
            features[f"BLOCK{current_stage_idx}_S{current_stride}"] = x

        if include_top:
            x = layers.GlobalAveragePooling2D(name="avg_pool", keepdims=True)(x)
            x = layers.Flatten()(x)
            x = layers.Dropout(rate=dropout_rate, name="head_dropout")(x)
            x = layers.Dense(
                classes, activation=classifier_activation, name="fc"
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
        self.block_fn = block_fn
        self.num_blocks = num_blocks
        self.include_preprocessing = include_preprocessing
        self.include_top = include_top
        self.pooling = pooling
        self.dropout_rate = dropout_rate
        self.classes = classes
        self.classifier_activation = classifier_activation
        self._weights = weights  # `self.weights` is been used internally

    @staticmethod
    def available_feature_keys():
        feature_keys = ["STEM_S2"]
        feature_keys.extend(
            [f"BLOCK{i}_S{j}" for i, j in zip(range(4), [4, 8, 16, 32])]
        )
        return feature_keys

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "block_fn": self.block_fn,
                "num_blocks": self.num_blocks,
                "input_shape": self.input_shape[1:],
                "include_preprocessing": self.include_preprocessing,
                "include_top": self.include_top,
                "pooling": self.pooling,
                "dropout_rate": self.dropout_rate,
                "classes": self.classes,
                "classifier_activation": self.classifier_activation,
                "weights": self._weights,
            }
        )
        return config

    def fix_config(self, config):
        unused_kwargs = ["block_fn", "num_blocks"]
        for k in unused_kwargs:
            config.pop(k, None)
        return config


"""
Model Definition
"""


class ResNet18(ResNet):
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
        weights: typing.Optional[str] = None,
        name: str = "ResNet18",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            "basic",
            [2, 2, 2, 2],
            input_tensor,
            input_shape,
            include_preprocessing,
            include_top,
            pooling,
            dropout_rate,
            classes,
            classifier_activation,
            weights,
            name=name,
            **kwargs,
        )


class ResNet34(ResNet):
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
        weights: typing.Optional[str] = None,
        name: str = "ResNet34",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            "basic",
            [3, 4, 6, 3],
            input_tensor,
            input_shape,
            include_preprocessing,
            include_top,
            pooling,
            dropout_rate,
            classes,
            classifier_activation,
            weights,
            name=name,
            **kwargs,
        )


class ResNet50(ResNet):
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
        weights: typing.Optional[str] = None,
        name: str = "ResNet50",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            "bottleneck",
            [3, 4, 6, 3],
            input_tensor,
            input_shape,
            include_preprocessing,
            include_top,
            pooling,
            dropout_rate,
            classes,
            classifier_activation,
            weights,
            name=name,
            **kwargs,
        )


class ResNet101(ResNet):
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
        weights: typing.Optional[str] = None,
        name: str = "ResNet101",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            "bottleneck",
            [3, 4, 23, 3],
            input_tensor,
            input_shape,
            include_preprocessing,
            include_top,
            pooling,
            dropout_rate,
            classes,
            classifier_activation,
            weights,
            name=name,
            **kwargs,
        )


class ResNet152(ResNet):
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
        weights: typing.Optional[str] = None,
        name: str = "ResNet152",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            "bottleneck",
            [3, 8, 36, 3],
            input_tensor,
            input_shape,
            include_preprocessing,
            include_top,
            pooling,
            dropout_rate,
            classes,
            classifier_activation,
            weights,
            name=name,
            **kwargs,
        )


add_model_to_registry(ResNet18, True)
add_model_to_registry(ResNet34, True)
add_model_to_registry(ResNet50, True)
add_model_to_registry(ResNet101, True)
add_model_to_registry(ResNet152, True)

import typing

import keras
from keras import backend
from keras import layers
from keras import utils
from keras.src.applications import imagenet_utils

from kimm.blocks import apply_conv2d_block
from kimm.models.feature_extractor import FeatureExtractor
from kimm.utils import add_model_to_registry


def apply_dense_layer(
    inputs, growth_rate, expansion_ratio=4.0, name="dense_layer"
):
    x = inputs
    x = layers.BatchNormalization(
        momentum=0.9, epsilon=1e-5, name=f"{name}_norm1"
    )(x)
    x = layers.ReLU()(x)
    x = apply_conv2d_block(
        x,
        int(growth_rate * expansion_ratio),
        1,
        1,
        activation="relu",
        name=f"{name}_conv1",
    )
    x = layers.Conv2D(
        growth_rate, 3, 1, padding="same", use_bias=False, name=f"{name}_conv2"
    )(x)
    return x


def apply_dense_block(
    inputs, num_layers, growth_rate, expansion_ratio=4.0, name="dense_block"
):
    x = inputs

    features = [x]
    for i in range(num_layers):
        new_features = layers.Concatenate()(features)
        new_features = apply_dense_layer(
            new_features,
            growth_rate,
            expansion_ratio,
            name=f"{name}_denselayer{i + 1}",
        )
        features.append(new_features)
    x = layers.Concatenate()(features)
    return x


def apply_dense_transition_block(
    inputs, output_channels, name="dense_transition_block"
):
    x = inputs
    x = layers.BatchNormalization(
        momentum=0.9, epsilon=1e-5, name=f"{name}_norm"
    )(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(
        output_channels, 1, 1, "same", use_bias=False, name=f"{name}_conv"
    )(x)
    x = layers.AveragePooling2D(2, 2, name=f"{name}_pool")(x)
    return x


class DenseNet(FeatureExtractor):
    def __init__(
        self,
        growth_rate: float = 32,
        num_blocks: typing.Sequence[int] = [6, 12, 24, 16],
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
        # default_size
        default_size = kwargs.pop("default_size", 224)

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

        # Stem block
        stem_channel = growth_rate * 2
        x = apply_conv2d_block(
            x, stem_channel, 7, 2, activation="relu", name="features_conv0"
        )
        x = layers.ZeroPadding2D(1, name="features_pad0")(x)
        x = layers.MaxPooling2D(3, 2, name="features_pool0")(x)
        features["STEM_S4"] = x

        # Blocks
        current_stride = 4
        input_channels = stem_channel
        for current_block_idx, num_layers in enumerate(num_blocks):
            x = apply_dense_block(
                x,
                num_layers,
                growth_rate,
                expansion_ratio=4.0,
                name=f"features_denseblock{current_block_idx + 1}",
            )
            input_channels = input_channels + num_layers * growth_rate
            if current_block_idx != len(num_blocks) - 1:
                current_stride *= 2
                x = apply_dense_transition_block(
                    x,
                    input_channels // 2,
                    name=f"features_transition{current_block_idx + 1}",
                )
                input_channels = input_channels // 2

            features[f"BLOCK{current_block_idx}_S{current_stride}"] = x

        # Final batch norm
        x = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5, name="features_norm5"
        )(x)
        x = layers.ReLU()(x)

        # Head
        if include_top:
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            x = layers.Dropout(rate=dropout_rate, name="head_dropout")(x)
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
        self.growth_rate = growth_rate
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
        feature_keys = ["STEM_S4"]
        feature_keys.extend(
            [f"BLOCK{i}_S{j}" for i, j in zip(range(4), [8, 16, 32, 32])]
        )
        return feature_keys

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "growth_rate": self.growth_rate,
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

    def fix_config(self, config: typing.Dict):
        unused_kwargs = ["growth_rate", "num_blocks"]
        for k in unused_kwargs:
            config.pop(k, None)
        return config


"""
Model Definition
"""


class DenseNet121(DenseNet):
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
        name: str = "DenseNet121",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            32,
            [6, 12, 24, 16],
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
            default_size=288,
            **kwargs,
        )


class DenseNet161(DenseNet):
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
        name: str = "DenseNet161",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            48,
            [6, 12, 36, 24],
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
            default_size=224,
            **kwargs,
        )


class DenseNet169(DenseNet):
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
        name: str = "DenseNet169",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            32,
            [6, 12, 32, 32],
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
            default_size=224,
            **kwargs,
        )


class DenseNet201(DenseNet):
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
        name: str = "DenseNet201",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            32,
            [6, 12, 48, 32],
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
            default_size=224,
            **kwargs,
        )


add_model_to_registry(DenseNet121, True)
add_model_to_registry(DenseNet161, True)
add_model_to_registry(DenseNet169, True)
add_model_to_registry(DenseNet201, True)

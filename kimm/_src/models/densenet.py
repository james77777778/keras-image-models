import typing

import keras
from keras import backend
from keras import layers

from kimm._src.blocks.conv2d import apply_conv2d_block
from kimm._src.kimm_export import kimm_export
from kimm._src.models.base_model import BaseModel
from kimm._src.utils.model_registry import add_model_to_registry


def apply_dense_layer(
    inputs, growth_rate, expansion_ratio=4.0, name="dense_layer"
):
    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3

    x = inputs
    x = layers.BatchNormalization(
        axis=channels_axis, momentum=0.9, epsilon=1e-5, name=f"{name}_norm1"
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
    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3

    x = inputs
    features = [x]
    for i in range(num_layers):
        new_features = layers.Concatenate(axis=channels_axis)(features)
        new_features = apply_dense_layer(
            new_features,
            growth_rate,
            expansion_ratio,
            name=f"{name}_denselayer{i + 1}",
        )
        features.append(new_features)
    x = layers.Concatenate(axis=channels_axis)(features)
    return x


def apply_dense_transition_block(
    inputs, output_channels, name="dense_transition_block"
):
    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3
    x = inputs
    x = layers.BatchNormalization(
        axis=channels_axis, momentum=0.9, epsilon=1e-5, name=f"{name}_norm"
    )(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(
        output_channels, 1, 1, "same", use_bias=False, name=f"{name}_conv"
    )(x)
    x = layers.AveragePooling2D(
        2, 2, data_format=backend.image_data_format(), name=f"{name}_pool"
    )(x)
    return x


@keras.saving.register_keras_serializable(package="kimm")
class DenseNet(BaseModel):
    available_feature_keys = [
        "STEM_S4",
        *[f"BLOCK{i}_S{j}" for i, j in zip(range(4), [8, 16, 32, 32])],
    ]

    def __init__(
        self,
        growth_rate: float = 32,
        num_blocks: typing.Sequence[int] = [6, 12, 24, 16],
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
            axis=channels_axis,
            momentum=0.9,
            epsilon=1e-5,
            name="features_norm5",
        )(x)
        x = layers.ReLU()(x)

        # Head
        x = self.build_head(x)

        super().__init__(inputs=inputs, outputs=x, features=features, **kwargs)

        # All references to `self` below this line
        self.growth_rate = growth_rate
        self.num_blocks = num_blocks

    def get_config(self):
        config = super().get_config()
        config.update(
            {"growth_rate": self.growth_rate, "num_blocks": self.num_blocks}
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


class DenseNetVariant(DenseNet):
    # Parameters
    growth_rate = None
    num_blocks = None
    default_size = None

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
        name: typing.Optional[str] = None,
        **kwargs,
    ):
        if type(self) is DenseNetVariant:
            raise NotImplementedError(
                f"Cannot instantiate base class: {self.__class__.__name__}. "
                "You should use its subclasses."
            )
        kwargs = self.fix_config(kwargs)
        super().__init__(
            growth_rate=self.growth_rate,
            num_blocks=self.num_blocks,
            input_tensor=input_tensor,
            input_shape=input_shape,
            include_preprocessing=include_preprocessing,
            include_top=include_top,
            pooling=pooling,
            dropout_rate=dropout_rate,
            classes=classes,
            classifier_activation=classifier_activation,
            weights=weights,
            name=name or str(self.__class__.__name__),
            default_size=self.default_size,
            **kwargs,
        )


@kimm_export(parent_path=["kimm.models", "kimm.models.densenet"])
class DenseNet121(DenseNetVariant):
    available_weights = [
        (
            "imagenet",
            DenseNet.default_origin,
            "densenet121_densenet121.ra_in1k.keras",
        )
    ]

    # Parameters
    growth_rate = 32
    num_blocks = [6, 12, 24, 16]
    default_size = 288


@kimm_export(parent_path=["kimm.models", "kimm.models.densenet"])
class DenseNet161(DenseNetVariant):
    available_weights = [
        (
            "imagenet",
            DenseNet.default_origin,
            "densenet161_densenet161.tv_in1k.keras",
        )
    ]

    # Parameters
    growth_rate = 48
    num_blocks = [6, 12, 36, 24]
    default_size = 224


@kimm_export(parent_path=["kimm.models", "kimm.models.densenet"])
class DenseNet169(DenseNetVariant):
    available_weights = [
        (
            "imagenet",
            DenseNet.default_origin,
            "densenet169_densenet169.tv_in1k.keras",
        )
    ]

    # Parameters
    growth_rate = 32
    num_blocks = [6, 12, 32, 32]
    default_size = 224


@kimm_export(parent_path=["kimm.models", "kimm.models.densenet"])
class DenseNet201(DenseNetVariant):
    available_weights = [
        (
            "imagenet",
            DenseNet.default_origin,
            "densenet201_densenet201.tv_in1k.keras",
        )
    ]

    # Parameters
    growth_rate = 32
    num_blocks = [6, 12, 48, 32]
    default_size = 224


add_model_to_registry(DenseNet121, "imagenet")
add_model_to_registry(DenseNet161, "imagenet")
add_model_to_registry(DenseNet169, "imagenet")
add_model_to_registry(DenseNet201, "imagenet")

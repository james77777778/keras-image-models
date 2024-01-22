import typing

import keras
from keras import backend
from keras import layers

from kimm.models import BaseModel
from kimm.utils import add_model_to_registry


def apply_xception_block(
    inputs,
    depth: int,
    output_channels: int,
    strides: int = 1,
    no_first_act=False,
    grow_first=True,
    name="xception_block",
):
    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3
    input_channels = inputs.shape[channels_axis]
    x = inputs
    residual = inputs

    current_layer_idx = 0
    for i in range(depth):
        if grow_first:
            c = output_channels
        else:
            c = input_channels if i < (depth - 1) else output_channels
        if not no_first_act or i > 0:
            x = layers.ReLU(name=f"{name}_rep_{current_layer_idx}")(x)
            current_layer_idx += 1
        x = layers.SeparableConv2D(
            c,
            3,
            1,
            "same",
            use_bias=False,
            name=f"{name}_rep_{current_layer_idx}",
        )(x)
        x = layers.BatchNormalization(
            axis=channels_axis,
            name=f"{name}_rep_{current_layer_idx + 1}",
        )(x)
        current_layer_idx += 2

    if strides != 1:
        x = layers.MaxPooling2D(
            3, strides, padding="same", name=f"{name}_rep_{current_layer_idx}"
        )(x)
        current_layer_idx += 1

    if output_channels != input_channels or strides != 1:
        residual = layers.Conv2D(
            output_channels,
            1,
            strides,
            use_bias=False,
            name=f"{name}_skipconv2d",
        )(residual)
        residual = layers.BatchNormalization(
            axis=channels_axis, name=f"{name}_skipbn"
        )(residual)

    x = layers.Add()([x, residual])
    return x


@keras.saving.register_keras_serializable(package="kimm")
class XceptionBase(BaseModel):
    available_feature_keys = [
        "STEM_S2",
        *[f"BLOCK{i}_S{j}" for i, j in zip(range(4), [4, 8, 16, 32])],
    ]

    def __init__(self, **kwargs):
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
            min_size=71,
        )
        x = inputs

        x = self.build_preprocessing(x, "-1_1")

        # Prepare feature extraction
        features = {}

        # Stem
        x = layers.Conv2D(32, 3, 2, use_bias=False, name="conv1")(x)
        x = layers.BatchNormalization(axis=channels_axis, name="bn1")(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(64, 3, 1, use_bias=False, name="conv2")(x)
        x = layers.BatchNormalization(axis=channels_axis, name="bn2")(x)
        x = layers.ReLU()(x)
        features["STEM_S2"] = x

        # Blocks
        x = apply_xception_block(x, 2, 128, 2, no_first_act=True, name="block1")
        features["BLOCK0_S4"] = x
        x = apply_xception_block(x, 2, 256, 2, name="block2")
        features["BLOCK1_S8"] = x
        x = apply_xception_block(x, 2, 728, 2, name="block3")

        x = apply_xception_block(x, 3, 728, 1, name="block4")
        x = apply_xception_block(x, 3, 728, 1, name="block5")
        x = apply_xception_block(x, 3, 728, 1, name="block6")
        x = apply_xception_block(x, 3, 728, 1, name="block7")

        x = apply_xception_block(x, 3, 728, 1, name="block8")
        x = apply_xception_block(x, 3, 728, 1, name="block9")
        x = apply_xception_block(x, 3, 728, 1, name="block10")
        x = apply_xception_block(x, 3, 728, 1, name="block11")
        features["BLOCK2_S16"] = x

        x = apply_xception_block(
            x, 2, 1024, 2, grow_first=False, name="block12"
        )

        x = layers.SeparableConv2D(
            1536, 3, 1, "same", use_bias=False, name="conv3"
        )(x)
        x = layers.BatchNormalization(axis=channels_axis, name="bn3")(x)
        x = layers.ReLU()(x)

        x = layers.SeparableConv2D(
            2048, 3, 1, "same", use_bias=False, name="conv4"
        )(x)
        x = layers.BatchNormalization(axis=channels_axis, name="bn4")(x)
        x = layers.ReLU()(x)
        features["BLOCK3_S32"] = x

        # Head
        x = self.build_head(x)

        super().__init__(inputs=inputs, outputs=x, features=features, **kwargs)

        # All references to `self` below this line

    def get_config(self):
        return super().get_config()

    def fix_config(self, config: typing.Dict):
        return config


"""
Model Definition
"""


class Xception(XceptionBase):
    available_weights = [
        (
            "imagenet",
            XceptionBase.default_origin,
            "xception.keras",
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
        name: str = "Xception",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
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
            default_size=299,
            **kwargs,
        )


add_model_to_registry(Xception, "imagenet")

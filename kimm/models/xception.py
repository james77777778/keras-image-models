import typing

import keras
from keras import layers
from keras import utils

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
    input_channels = inputs.shape[-1]
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
            name=f"{name}_skipbn",
        )(residual)

    x = layers.Add()([x, residual])
    return x


class XceptionBase(BaseModel):
    def __init__(self, **kwargs):
        parsed_kwargs = self.parse_kwargs(kwargs)
        img_input = self.determine_input_tensor(
            parsed_kwargs["input_tensor"],
            parsed_kwargs["input_shape"],
            parsed_kwargs["default_size"],
            min_size=71,
        )
        x = img_input

        if parsed_kwargs["include_preprocessing"]:
            x = self.build_preprocessing(x, "-1_1")

        # Prepare feature extraction
        features = {}

        # Stem
        x = layers.Conv2D(32, 3, 2, use_bias=False, name="conv1")(x)
        x = layers.BatchNormalization(name="bn1")(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(64, 3, 1, use_bias=False, name="conv2")(x)
        x = layers.BatchNormalization(name="bn2")(x)
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
        x = layers.BatchNormalization(name="bn3")(x)
        x = layers.ReLU()(x)

        x = layers.SeparableConv2D(
            2048, 3, 1, "same", use_bias=False, name="conv4"
        )(x)
        x = layers.BatchNormalization(name="bn4")(x)
        x = layers.ReLU()(x)
        features["BLOCK3_S32"] = x

        # Head
        if parsed_kwargs["include_top"]:
            x = self.build_top(
                x,
                parsed_kwargs["classes"],
                parsed_kwargs["classifier_activation"],
                parsed_kwargs["dropout_rate"],
            )
        else:
            if parsed_kwargs["pooling"] == "avg":
                x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            elif parsed_kwargs["pooling"] == "max":
                x = layers.GlobalMaxPooling2D(name="max_pool")(x)

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        if parsed_kwargs["input_tensor"] is not None:
            inputs = utils.get_source_inputs(parsed_kwargs["input_tensor"])
        else:
            inputs = img_input

        super().__init__(inputs=inputs, outputs=x, features=features, **kwargs)

        # All references to `self` below this line
        self.add_references(parsed_kwargs)

    @staticmethod
    def available_feature_keys():
        feature_keys = ["STEM_S2"]
        feature_keys.extend(
            [f"BLOCK{i}_S{j}" for i, j in zip(range(4), [4, 8, 16, 32])]
        )
        return feature_keys

    def get_config(self):
        return super().get_config()

    def fix_config(self, config: typing.Dict):
        return config


"""
Model Definition
"""


class Xception(XceptionBase):
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

import typing

import keras
from keras import backend
from keras import layers

from kimm._src.blocks.conv2d import apply_conv2d_block
from kimm._src.kimm_export import kimm_export
from kimm._src.models.base_model import BaseModel
from kimm._src.utils.model_registry import add_model_to_registry


def apply_basic_block(
    inputs,
    output_channels: int,
    strides: int = 1,
    activation="relu",
    name="basic_block",
):
    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3
    input_channels = inputs.shape[channels_axis]
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
    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3
    input_channels = inputs.shape[channels_axis]
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


@keras.saving.register_keras_serializable(package="kimm")
class ResNet(BaseModel):
    available_feature_keys = [
        "STEM_S2",
        *[f"BLOCK{i}_S{j}" for i, j in zip(range(4), [4, 8, 16, 32])],
    ]

    def __init__(
        self,
        block_fn: typing.Literal["basic", "bottleneck"],
        num_blocks: typing.Sequence[int],
        input_tensor=None,
        **kwargs,
    ):
        if block_fn not in ("basic", "bottleneck"):
            raise ValueError(
                "`block_fn` must be one of ('basic', 'bottelneck'). "
                f"Received: block_fn={block_fn}"
            )

        self.set_properties(kwargs)
        inputs = self.determine_input_tensor(
            input_tensor,
            self._input_shape,
            self._default_size,
        )
        x = inputs

        x = self.build_preprocessing(x, "imagenet")

        # Prepare feature extraction
        features = {}

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

        # Head
        x = self.build_head(x)

        super().__init__(inputs=inputs, outputs=x, features=features, **kwargs)

        # All references to `self` below this line
        self.block_fn = block_fn
        self.num_blocks = num_blocks

    def get_config(self):
        config = super().get_config()
        config.update(
            {"block_fn": self.block_fn, "num_blocks": self.num_blocks}
        )
        return config

    def fix_config(self, config):
        unused_kwargs = ["block_fn", "num_blocks"]
        for k in unused_kwargs:
            config.pop(k, None)
        return config


# Model Definition


class ResNetVariant(ResNet):
    # Parameters
    block_fn = None
    num_blocks = None

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
        if type(self) is ResNetVariant:
            raise NotImplementedError(
                f"Cannot instantiate base class: {self.__class__.__name__}. "
                "You should use its subclasses."
            )
        kwargs = self.fix_config(kwargs)
        super().__init__(
            block_fn=self.block_fn,
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
            **kwargs,
        )


@kimm_export(parent_path=["kimm.models", "kimm.models.resnet"])
class ResNet18(ResNetVariant):
    available_weights = [
        (
            "imagenet",
            ResNet.default_origin,
            "resnet18_resnet18.a1_in1k.keras",
        )
    ]

    # Parameters
    block_fn = "basic"
    num_blocks = [2, 2, 2, 2]


@kimm_export(parent_path=["kimm.models", "kimm.models.resnet"])
class ResNet34(ResNetVariant):
    available_weights = [
        (
            "imagenet",
            ResNet.default_origin,
            "resnet34_resnet34.a1_in1k.keras",
        )
    ]

    # Parameters
    block_fn = "basic"
    num_blocks = [3, 4, 6, 3]


@kimm_export(parent_path=["kimm.models", "kimm.models.resnet"])
class ResNet50(ResNetVariant):
    available_weights = [
        (
            "imagenet",
            ResNet.default_origin,
            "resnet50_resnet50.a1_in1k.keras",
        )
    ]

    # Parameters
    block_fn = "bottleneck"
    num_blocks = [3, 4, 6, 3]


@kimm_export(parent_path=["kimm.models", "kimm.models.resnet"])
class ResNet101(ResNetVariant):
    available_weights = [
        (
            "imagenet",
            ResNet.default_origin,
            "resnet101_resnet101.a1_in1k.keras",
        )
    ]

    # Parameters
    block_fn = "bottleneck"
    num_blocks = [3, 4, 23, 3]


@kimm_export(parent_path=["kimm.models", "kimm.models.resnet"])
class ResNet152(ResNetVariant):
    available_weights = [
        (
            "imagenet",
            ResNet.default_origin,
            "resnet152_resnet152.a1_in1k.keras",
        )
    ]

    # Parameters
    block_fn = "bottleneck"
    num_blocks = [3, 8, 36, 3]


add_model_to_registry(ResNet18, "imagenet")
add_model_to_registry(ResNet34, "imagenet")
add_model_to_registry(ResNet50, "imagenet")
add_model_to_registry(ResNet101, "imagenet")
add_model_to_registry(ResNet152, "imagenet")

import pathlib
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
        input_tensor: typing.Optional[keras.KerasTensor] = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        dropout_rate: float = 0.0,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[typing.Union[str, pathlib.Path]] = "imagenet",
        name: typing.Optional[str] = None,
        feature_extractor: bool = False,
        feature_keys: typing.Optional[typing.Sequence[str]] = None,
        **kwargs,
    ):
        """Instantiates the ResNet architecture.

        Reference:
        - [Deep Residual Learning for Image Recognition (CVPR 2015)]
        (https://arxiv.org/abs/1512.03385)
        - [ResNet strikes back: An improved training procedure in timm](https://arxiv.org/abs/2110.00476)

        Args:
            input_tensor: An optional `keras.KerasTensor` specifying the input.
            input_shape: An optional sequence of ints specifying the input
                shape.
            include_preprocessing: Whether to include preprocessing. Defaults
                to `True`.
            include_top: Whether to include prediction head. Defaults
                to `True`.
            pooling: An optional `str` specifying the pooling type on top of
                the model. This argument only takes effect if
                `include_top=False`. Available values are `"avg"` and `"max"`
                which correspond to `GlobalAveragePooling2D` and
                `GlobalMaxPooling2D`, respectively. Defaults to `None`.
            dropout_rate: A `float` specifying the dropout rate in prediction
                head. This argument only takes effect if `include_top=True`.
                Defaults to `0.0`.
            classes: An `int` specifying the number of classes. Defaults to
                `1000` for ImageNet.
            classifier_activation: A `str` specifying the activation
                function of the final output. Defaults to `"softmax"`.
            weights: An optional `str` or `pathlib.Path` specifying the name,
                url or path of the pretrained weights. Defaults to `"imagenet"`.
            name: An optional `str` specifying the name of the model. If not
                specified, it will be the class name. Defaults to `None`.
            feature_extractor: Whether to enable feature extraction. If `True`,
                the outputs will be a `dict` that keys are feature names and
                values are feature maps. Defaults to `False`.
            feature_keys: An optional sequence of strings specifying the
                selected feature names. This argument only takes effect if
                `feature_extractor=True`. Defaults to `None`.

        Returns:
            A `keras.Model` instance.
        """
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
            feature_extractor=feature_extractor,
            feature_keys=feature_keys,
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

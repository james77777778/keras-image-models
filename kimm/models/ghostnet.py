import typing

import keras
from keras import backend
from keras import layers
from keras import ops

from kimm.blocks import apply_conv2d_block
from kimm.blocks import apply_se_block
from kimm.models import BaseModel
from kimm.utils import add_model_to_registry
from kimm.utils import make_divisible

DEFAULT_CONFIG = [
    # k, t, c, SE, s
    # stage1
    [
        [3, 16, 16, 0, 1],
    ],
    # stage2
    [
        [3, 48, 24, 0, 2],
    ],
    [
        [3, 72, 24, 0, 1],
    ],
    # stage3
    [
        [5, 72, 40, 0.25, 2],
    ],
    [
        [5, 120, 40, 0.25, 1],
    ],
    # stage4
    [
        [3, 240, 80, 0, 2],
    ],
    [
        [3, 200, 80, 0, 1],
        [3, 184, 80, 0, 1],
        [3, 184, 80, 0, 1],
        [3, 480, 112, 0.25, 1],
        [3, 672, 112, 0.25, 1],
    ],
    # stage5
    [
        [5, 672, 160, 0.25, 2],
    ],
    [
        [5, 960, 160, 0, 1],
        [5, 960, 160, 0.25, 1],
        [5, 960, 160, 0, 1],
        [5, 960, 160, 0.25, 1],
    ],
]


def apply_ghost_block(
    inputs,
    output_channels: int,
    expand_ratio: float = 2.0,
    kernel_size: int = 1,
    depthwise_kernel_size: int = 3,
    strides: int = 1,
    activation="relu",
    name="ghost_block",
):
    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3
    hidden_channels_1 = int(ops.ceil(output_channels / expand_ratio))
    hidden_channels_2 = int(hidden_channels_1 * (expand_ratio - 1.0))

    x = inputs
    x1 = apply_conv2d_block(
        x,
        hidden_channels_1,
        kernel_size,
        strides,
        activation=activation,
        name=f"{name}_primary_conv",
    )
    x2 = apply_conv2d_block(
        x1,
        hidden_channels_2,
        depthwise_kernel_size,
        1,
        activation=activation,
        use_depthwise=True,
        name=f"{name}_cheap_operation",
    )
    out = layers.Concatenate(axis=channels_axis, name=f"{name}")([x1, x2])
    if channels_axis == -1:
        return out[..., :output_channels]
    else:
        return out[:, :output_channels, ...]


def apply_ghost_block_v2(
    inputs,
    output_channels: int,
    expand_ratio: float = 2.0,
    kernel_size: int = 1,
    depthwise_kernel_size: int = 3,
    strides: int = 1,
    activation="relu",
    name="ghost_block_v2",
):
    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3
    if backend.image_data_format() == "channels_last":
        output_axis = (-3, -2)
    else:
        output_axis = (-2, -1)

    hidden_channels_1 = int(ops.ceil(output_channels / expand_ratio))
    hidden_channels_2 = int(hidden_channels_1 * (expand_ratio - 1.0))

    x = inputs
    residual = inputs
    x1 = apply_conv2d_block(
        x,
        hidden_channels_1,
        kernel_size,
        strides,
        activation=activation,
        name=f"{name}_primary_conv",
    )
    x2 = apply_conv2d_block(
        x1,
        hidden_channels_2,
        depthwise_kernel_size,
        1,
        activation=activation,
        use_depthwise=True,
        name=f"{name}_cheap_operation",
    )
    out = layers.Concatenate(axis=channels_axis, name=f"{name}_concat")(
        [x1, x2]
    )

    residual = layers.AveragePooling2D(
        2, 2, data_format=backend.image_data_format(), name=f"{name}_avg_pool"
    )(residual)
    residual = apply_conv2d_block(
        residual,
        output_channels,
        kernel_size,
        strides,
        name=f"{name}_short_conv1",
    )
    residual = apply_conv2d_block(
        residual,
        output_channels,
        (1, 5),
        1,
        use_depthwise=True,
        name=f"{name}_short_conv2",
    )
    residual = apply_conv2d_block(
        residual,
        output_channels,
        (5, 1),
        1,
        use_depthwise=True,
        name=f"{name}_short_conv3",
    )
    residual = layers.Activation("sigmoid", name=f"{name}_gate")(residual)
    # TODO: support dynamic shape
    residual = layers.Resizing(
        out.shape[output_axis[0]], out.shape[output_axis[1]], "nearest"
    )(residual)
    if channels_axis == -1:
        out = out[..., :output_channels]
    else:
        out = out[:, :output_channels, ...]
    out = layers.Multiply(name=name)([out, residual])
    return out


def apply_ghost_bottleneck(
    inputs,
    hidden_channels: int,
    output_channels: int,
    depthwise_kernel_size: int = 3,
    strides: int = 1,
    se_ratio: float = 0.0,
    activation="relu",
    use_attention=False,  # GhostNetV2
    name="ghost_bottlenect",
):
    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3
    input_channels = inputs.shape[channels_axis]
    has_se = se_ratio is not None and se_ratio > 0.0

    x = inputs
    shortcut = inputs
    # point-wise
    if use_attention:
        x = apply_ghost_block_v2(
            x, hidden_channels, activation=activation, name=f"{name}_ghost1"
        )
    else:
        x = apply_ghost_block(
            x, hidden_channels, activation=activation, name=f"{name}_ghost1"
        )
    # depthwise
    if strides > 1:
        x = apply_conv2d_block(
            x,
            hidden_channels,
            depthwise_kernel_size,
            strides,
            use_depthwise=True,
            name=f"{name}_conv_dw",
        )
    # squeeze-and-excitation
    if has_se:
        x = apply_se_block(
            x,
            se_ratio,
            gate_activation="hard_sigmoid",
            make_divisible_number=4,
            name=f"{name}_se",
        )
    # point-wise
    x = apply_ghost_block(
        x, output_channels, activation=None, name=f"{name}_ghost2"
    )
    # shortcut
    if input_channels != output_channels or strides > 1:
        shortcut = apply_conv2d_block(
            shortcut,
            input_channels,
            depthwise_kernel_size,
            strides,
            activation=None,
            use_depthwise=True,
            name=f"{name}_shortcut1",
        )
        shortcut = apply_conv2d_block(
            shortcut,
            output_channels,
            1,
            1,
            activation=None,
            name=f"{name}_shortcut2",
        )
    out = layers.Add(name=name)([x, shortcut])
    return out


@keras.saving.register_keras_serializable(package="kimm")
class GhostNet(BaseModel):
    available_feature_keys = [
        "STEM_S2",
        *[
            f"BLOCK{i}_S{j}"
            for i, j in zip(range(9), [2, 4, 4, 8, 8, 16, 16, 32, 32])
        ],
    ]

    def __init__(
        self,
        width: float = 1.0,
        config: typing.Union[str, typing.List] = "default",
        version: typing.Literal["v1", "v2"] = "v1",
        **kwargs,
    ):
        kwargs["weights_url"] = self.get_weights_url(kwargs["weights"])

        _available_configs = ["default"]
        if config == "default":
            _config = DEFAULT_CONFIG
        else:
            raise ValueError(
                f"config must be one of {_available_configs} using string. "
                f"Received: config={config}"
            )
        if version not in ("v1", "v2"):
            raise ValueError(
                "`version` must be one of ('v1', 'v2'). "
                f"Received version={version}"
            )

        input_tensor = kwargs.pop("input_tensor", None)
        self.set_properties(kwargs)
        inputs = self.determine_input_tensor(
            input_tensor,
            self._input_shape,
            self._default_size,
            require_flatten=self._include_top,
            static_shape=True if version == "v2" else False,
        )
        x = inputs

        x = self.build_preprocessing(x, "imagenet")

        # Prepare feature extraction
        features = {}

        # stem
        stem_channels = make_divisible(16 * width, 4)
        x = apply_conv2d_block(
            x, stem_channels, 3, 2, activation="relu", name="conv_stem"
        )
        features["STEM_S2"] = x

        # blocks
        total_layer_idx = 0
        current_stride = 2
        for current_block_idx, cfg in enumerate(_config):
            for current_layer_idx, (k, e, c, se, s) in enumerate(cfg):
                output_channels = make_divisible(c * width, 4)
                hidden_channels = make_divisible(e * width, 4)
                use_attention = False
                if version == "v2" and total_layer_idx > 1:
                    use_attention = True
                name = f"blocks_{current_block_idx}_{current_layer_idx}"
                x = apply_ghost_bottleneck(
                    x,
                    hidden_channels,
                    output_channels,
                    k,
                    s,
                    se_ratio=se,
                    use_attention=use_attention,
                    name=name,
                )
                total_layer_idx += 1
            current_stride *= s
            features[f"BLOCK{current_block_idx}_S{current_stride}"] = x
        # post stages conv block
        output_channels = make_divisible(e * width, 4)
        x = apply_conv2d_block(
            x,
            output_channels,
            1,
            activation="relu",
            name=f"blocks_{current_block_idx+1}",
        )

        # Head
        x = self.build_head(x)

        super().__init__(inputs=inputs, outputs=x, features=features, **kwargs)

        # All references to `self` below this line
        self.width = width
        self.config = config
        self.version = version

    def build_top(self, inputs, classes, classifier_activation, dropout_rate):
        x = layers.GlobalAveragePooling2D(name="avg_pool", keepdims=True)(
            inputs
        )
        x = layers.Conv2D(
            1280, 1, 1, use_bias=True, activation="relu", name="conv_head"
        )(x)
        x = layers.Flatten()(x)
        x = layers.Dropout(rate=dropout_rate, name="conv_head_dropout")(x)
        x = layers.Dense(
            classes, activation=classifier_activation, name="classifier"
        )(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "width": self.width,
                "config": self.config,
                "version": self.version,
            }
        )
        return config

    def fix_config(self, config):
        unused_kwargs = ["width", "config", "version"]
        for k in unused_kwargs:
            config.pop(k, None)
        return config


"""
Model Definition
"""


class GhostNet050(GhostNet):
    available_weights = []

    def __init__(
        self,
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        dropout_rate: float = 0.2,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = None,
        config: typing.Union[str, typing.List] = "default",
        name: str = "GhostNet050",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            0.5,
            config,
            "v1",
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


class GhostNet100(GhostNet):
    available_weights = [
        (
            "imagenet",
            GhostNet.default_origin,
            "ghostnet100_ghostnet_100.keras",
        )
    ]

    def __init__(
        self,
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        dropout_rate: float = 0.2,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = "imagenet",
        config: typing.Union[str, typing.List] = "default",
        name: str = "GhostNet100",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            1.0,
            config,
            "v1",
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


class GhostNet130(GhostNet):
    available_weights = []

    def __init__(
        self,
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        dropout_rate: float = 0.2,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = None,
        config: typing.Union[str, typing.List] = "default",
        name: str = "GhostNet130",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            1.3,
            config,
            "v1",
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


class GhostNet100V2(GhostNet):
    available_weights = [
        (
            "imagenet",
            GhostNet.default_origin,
            "ghostnet100v2_ghostnetv2_100.keras",
        )
    ]

    def __init__(
        self,
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        dropout_rate: float = 0.2,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = "imagenet",
        config: typing.Union[str, typing.List] = "default",
        name: str = "GhostNet100V2",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            1.0,
            config,
            "v2",
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


class GhostNet130V2(GhostNet):
    available_weights = [
        (
            "imagenet",
            GhostNet.default_origin,
            "ghostnet130v2_ghostnetv2_130.keras",
        )
    ]

    def __init__(
        self,
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        dropout_rate: float = 0.2,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = "imagenet",
        config: typing.Union[str, typing.List] = "default",
        name: str = "GhostNet130V2",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            1.3,
            config,
            "v2",
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


class GhostNet160V2(GhostNet):
    available_weights = [
        (
            "imagenet",
            GhostNet.default_origin,
            "ghostnet160v2_ghostnetv2_160.keras",
        )
    ]

    def __init__(
        self,
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        dropout_rate: float = 0.2,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = "imagenet",
        config: typing.Union[str, typing.List] = "default",
        name: str = "GhostNet160V2",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            1.6,
            config,
            "v2",
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


add_model_to_registry(GhostNet050)
add_model_to_registry(GhostNet100, "imagenet")
add_model_to_registry(GhostNet130)
add_model_to_registry(GhostNet100V2, "imagenet")
add_model_to_registry(GhostNet130V2, "imagenet")
add_model_to_registry(GhostNet160V2, "imagenet")

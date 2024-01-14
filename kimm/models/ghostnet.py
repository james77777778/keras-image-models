import typing

import keras
from keras import backend
from keras import layers
from keras import ops
from keras import utils
from keras.src.applications import imagenet_utils

from kimm.blocks import apply_conv2d_block
from kimm.blocks import apply_se_block
from kimm.models.feature_extractor import FeatureExtractor
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
    out = layers.Concatenate(name=f"{name}")([x1, x2])
    return out[..., :output_channels]


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
    out = layers.Concatenate(name=f"{name}_concat")([x1, x2])

    residual = layers.AveragePooling2D(2, 2, name=f"{name}_avg_pool")(residual)
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
    residual = layers.Resizing(out.shape[-3], out.shape[-2], "nearest")(
        residual
    )
    out = out[..., :output_channels]
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
    input_channels = inputs.shape[-1]
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


class GhostNet(FeatureExtractor):
    def __init__(
        self,
        width: float = 1.0,
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        dropout_rate: float = 0.2,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = None,  # TODO: imagenet
        config: typing.Union[str, typing.List] = "default",
        version: str = "v1",
        **kwargs,
    ):
        if config == "default":
            config = DEFAULT_CONFIG
        if version not in ("v1", "v2"):
            raise ValueError(
                "`version` must be one of ('v1', 'v2'). "
                f"Received version={version}"
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
        stem_channels = make_divisible(16 * width, 4)
        x = apply_conv2d_block(
            x, stem_channels, 3, 2, activation="relu", name="conv_stem"
        )
        features["STEM_S2"] = x

        # blocks
        total_layer_idx = 0
        current_stride = 2
        for current_block_idx, cfg in enumerate(config):
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

        if include_top:
            x = layers.GlobalAveragePooling2D(name="avg_pool", keepdims=True)(x)
            x = layers.Conv2D(1280, 1, 1, use_bias=True, name="conv_head")(x)
            x = layers.ReLU(name="conv_head_relu")(x)
            x = layers.Flatten()(x)
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
        self.include_preprocessing = include_preprocessing
        self.include_top = include_top
        self.pooling = pooling
        self.dropout_rate = dropout_rate
        self.classes = classes
        self.classifier_activation = classifier_activation
        self._weights = weights  # `self.weights` is been used internally
        self.config = config
        self.version = version

    @staticmethod
    def available_feature_keys():
        feature_keys = ["STEM_S2"]
        feature_keys.extend(
            [
                f"BLOCK{i}_S{j}"
                for i, j in zip(range(9), [2, 4, 4, 8, 8, 16, 16, 32, 32])
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
                "version": self.version,
            }
        )
        return config


"""
Model Definition
"""


class GhostNet050(GhostNet):
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
        super().__init__(
            0.5,
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


class GhostNet100(GhostNet):
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
        weights: typing.Optional[str] = None,  # TODO: imagenet
        config: typing.Union[str, typing.List] = "default",
        name: str = "GhostNet100",
        **kwargs,
    ):
        super().__init__(
            1.0,
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


class GhostNet130(GhostNet):
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
        super().__init__(
            1.3,
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


class GhostNet100V2(GhostNet):
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
        weights: typing.Optional[str] = None,  # TODO: imagenet
        config: typing.Union[str, typing.List] = "default",
        name: str = "GhostNet100V2",
        **kwargs,
    ):
        super().__init__(
            1.0,
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
            "v2",
            name=name,
            **kwargs,
        )


class GhostNet130V2(GhostNet):
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
        weights: typing.Optional[str] = None,  # TODO: imagenet
        config: typing.Union[str, typing.List] = "default",
        name: str = "GhostNet130V2",
        **kwargs,
    ):
        super().__init__(
            1.3,
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
            "v2",
            name=name,
            **kwargs,
        )


class GhostNet160V2(GhostNet):
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
        weights: typing.Optional[str] = None,  # TODO: imagenet
        config: typing.Union[str, typing.List] = "default",
        name: str = "GhostNet160V2",
        **kwargs,
    ):
        super().__init__(
            1.6,
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
            "v2",
            name=name,
            **kwargs,
        )


add_model_to_registry(GhostNet050, False)
add_model_to_registry(GhostNet100, True)
add_model_to_registry(GhostNet130, True)
add_model_to_registry(GhostNet100V2, True)
add_model_to_registry(GhostNet130V2, True)
add_model_to_registry(GhostNet160V2, True)

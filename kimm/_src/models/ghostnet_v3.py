import pathlib
import typing
import warnings

import keras
from keras import backend
from keras import layers
from keras import ops

from kimm._src.blocks.conv2d import apply_conv2d_block
from kimm._src.blocks.squeeze_and_excitation import apply_se_block
from kimm._src.kimm_export import kimm_export
from kimm._src.layers.reparameterizable_conv2d import ReparameterizableConv2D
from kimm._src.models.base_model import BaseModel
from kimm._src.utils.make_divisble import make_divisible
from kimm._src.utils.model_registry import add_model_to_registry

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


def apply_short_block(
    inputs,
    output_channels: int,
    kernel_size: int = 1,
    strides: int = 1,
    name="short_block",
):
    x = inputs
    x = apply_conv2d_block(
        x,
        output_channels,
        kernel_size,
        strides,
        activation=None,
        name=f"{name}_0",
    )
    x = apply_conv2d_block(
        x,
        output_channels,
        (1, 5),
        1,
        activation=None,
        use_depthwise=True,
        padding="same",
        name=f"{name}_1",
    )
    x = apply_conv2d_block(
        x,
        output_channels,
        (5, 1),
        1,
        activation=None,
        use_depthwise=True,
        padding="same",
        name=f"{name}_2",
    )
    return x


def apply_ghost_block_v3(
    inputs,
    output_channels: int,
    expand_ratio: float = 2.0,
    kernel_size: int = 1,
    depthwise_kernel_size: int = 3,
    strides: int = 1,
    activation="relu",
    mode="ori",
    reparameterized: bool = False,
    name="ghost_block_v3",
):
    assert mode in ("ori", "ori_shortcut_mul_conv15")

    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3
    hidden_channels_1 = int(ops.ceil(output_channels / expand_ratio))
    hidden_channels_2 = int(hidden_channels_1 * (expand_ratio - 1.0))
    input_channels = inputs.shape[channels_axis]
    has_skip1 = input_channels == hidden_channels_1 and strides == 1
    has_skip2 = hidden_channels_1 == hidden_channels_2
    has_scale1 = kernel_size > 1
    has_scale2 = depthwise_kernel_size > 1

    x = inputs
    residual = inputs

    x1 = ReparameterizableConv2D(
        hidden_channels_1,
        kernel_size,
        strides,
        has_skip=has_skip1,
        has_scale=has_scale1,
        branch_size=3,
        reparameterized=reparameterized,
        activation=activation,
        name=f"{name}_primary_conv",
    )(x)
    x2 = ReparameterizableConv2D(
        hidden_channels_2,
        depthwise_kernel_size,
        1,
        has_skip=has_skip2,
        has_scale=has_scale2,
        use_depthwise=True,
        branch_size=3,
        reparameterized=reparameterized,
        activation=activation,
        name=f"{name}_cheap_operation",
    )(x1)
    out = layers.Concatenate(axis=channels_axis)([x1, x2])

    if mode == "ori_shortcut_mul_conv15":
        if channels_axis == -1:
            out = out[..., :output_channels]
            h, w = out.shape[-3], out.shape[-2]
        else:
            out = out[:, :output_channels, ...]
            h, w = out.shape[-2], out.shape[-1]
        residual = layers.AveragePooling2D(2, 2)(x)
        residual = apply_short_block(
            residual,
            output_channels,
            kernel_size,
            strides,
            name=f"{name}_short_conv",
        )
        residual = layers.Activation("sigmoid")(residual)
        residual = ops.image.resize(
            residual,
            size=(h, w),
            interpolation="nearest",
            data_format=backend.image_data_format(),
        )
        out = layers.Multiply()([out, residual])

    return out


def apply_ghost_bottleneck_v3(
    inputs,
    hidden_channels: int,
    output_channels: int,
    depthwise_kernel_size: int = 3,
    strides: int = 1,
    se_ratio: float = 0.0,
    activation="relu",
    pw_ghost_mode="ori",
    reparameterized: bool = False,
    name="ghost_bottlenect",
):
    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3
    input_channels = inputs.shape[channels_axis]
    has_skip = strides == 1
    has_scale = depthwise_kernel_size > 1
    has_se = se_ratio is not None and se_ratio > 0.0

    x = inputs
    shortcut = inputs

    # Point-wise expansion
    x = apply_ghost_block_v3(
        x,
        hidden_channels,
        activation=activation,
        mode=pw_ghost_mode,
        reparameterized=reparameterized,
        name=f"{name}_ghost1",
    )

    # Depth-wise
    if strides > 1:
        x = ReparameterizableConv2D(
            hidden_channels,
            depthwise_kernel_size,
            strides=strides,
            has_skip=has_skip,
            has_scale=has_scale,
            use_depthwise=True,
            branch_size=3,
            reparameterized=reparameterized,
            activation=None,
            name=f"{name}_conv_dw",
        )(x)

    # Squeeze-and-excitation
    if has_se:
        x = apply_se_block(
            x,
            se_ratio,
            gate_activation="hard_sigmoid",
            make_divisible_number=4,
            name=f"{name}_se",
        )

    # Point-wise
    x = apply_ghost_block_v3(
        x,
        output_channels,
        activation=None,
        mode="ori",
        reparameterized=reparameterized,
        name=f"{name}_ghost2",
    )

    # Shortcut
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
            padding="valid",
            name=f"{name}_shortcut2",
        )

    out = layers.Add(name=name)([x, shortcut])
    return out


@keras.saving.register_keras_serializable(package="kimm")
class GhostNetV3(BaseModel):
    # Updated weights: use ReparameterizableConv2D
    default_origin = "https://github.com/james77777778/keras-image-models/releases/download/0.1.2/"
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
        reparameterized: bool = False,
        input_tensor=None,
        **kwargs,
    ):
        _available_configs = ["default"]
        if config == "default":
            _config = DEFAULT_CONFIG
        else:
            raise ValueError(
                f"config must be one of {_available_configs} using string. "
                f"Received: config={config}"
            )

        self.set_properties(kwargs)
        inputs = self.determine_input_tensor(
            input_tensor,
            self._input_shape,
            self._default_size,
            require_flatten=self._include_top,
            static_shape=True,
        )
        x = inputs

        x = self.build_preprocessing(x, "imagenet")

        # Prepare feature extraction
        features = {}

        # Stem
        stem_channels = make_divisible(16 * width, 4)
        x = apply_conv2d_block(
            x, stem_channels, 3, 2, activation="relu", name="conv_stem"
        )
        features["STEM_S2"] = x

        # Blocks
        total_layer_idx = 0
        current_stride = 2
        for current_block_idx, cfg in enumerate(_config):
            for current_layer_idx, (k, e, c, se, s) in enumerate(cfg):
                output_channels = make_divisible(c * width, 4)
                hidden_channels = make_divisible(e * width, 4)
                pw_ghost_mode = (
                    "ori" if total_layer_idx <= 1 else "ori_shortcut_mul_conv15"
                )
                name = f"blocks_{current_block_idx}_{current_layer_idx}"
                x = apply_ghost_bottleneck_v3(
                    x,
                    hidden_channels,
                    output_channels,
                    k,
                    s,
                    se_ratio=se,
                    pw_ghost_mode=pw_ghost_mode,
                    reparameterized=reparameterized,
                    name=name,
                )
                total_layer_idx += 1
            current_stride *= s
            features[f"BLOCK{current_block_idx}_S{current_stride}"] = x

        # Last block
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
        self.reparameterized = reparameterized

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
                "reparameterized": self.reparameterized,
            }
        )
        return config

    def fix_config(self, config):
        unused_kwargs = ["width", "config"]
        for k in unused_kwargs:
            config.pop(k, None)
        return config

    def get_reparameterized_model(self):
        config = self.get_config()
        config["reparameterized"] = True
        config["weights"] = None
        model = GhostNetV3(**config)
        for layer, rep_layer in zip(self.layers, model.layers):
            if hasattr(layer, "get_reparameterized_weights"):
                kernel, bias = layer.get_reparameterized_weights()
                rep_layer.reparameterized_conv2d.kernel.assign(kernel)
                rep_layer.reparameterized_conv2d.bias.assign(bias)
            else:
                for weight, target_weight in zip(
                    layer.weights, rep_layer.weights
                ):
                    target_weight.assign(weight)
        return model


# Model Definition


class GhostNetV3Variant(GhostNetV3):
    # Parameters
    width = None
    config = None

    def __init__(
        self,
        reparameterized: bool = False,
        input_tensor: typing.Optional[keras.KerasTensor] = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        dropout_rate: float = 0.0,  # Defaults to 0.0
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[typing.Union[str, pathlib.Path]] = "imagenet",
        name: typing.Optional[str] = None,
        feature_extractor: bool = False,
        feature_keys: typing.Optional[typing.Sequence[str]] = None,
        **kwargs,
    ):
        """Instantiates the GhostNetV3 architecture.

        Reference:
        - [GhostNetV3: Exploring the Training Strategies for Compact Models
        (arXiv 2024)](https://arxiv.org/abs/2404.11202)

        Args:
            reparameterized: Whether to instantiate the model with
                reparameterized state. Defaults to `False`. Note that
                pretrained weights are only available with
                `reparameterized=False`.
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
        if type(self) is GhostNetV3Variant:
            raise NotImplementedError(
                f"Cannot instantiate base class: {self.__class__.__name__}. "
                "You should use its subclasses."
            )
        kwargs = self.fix_config(kwargs)
        if len(getattr(self, "available_weights", [])) == 0:
            warnings.warn(
                f"{self.__class__.__name__} doesn't have pretrained weights "
                f"for '{weights}'."
            )
            weights = None
        super().__init__(
            width=self.width,
            config=self.config,
            reparameterized=reparameterized,
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


@kimm_export(parent_path=["kimm.models", "kimm.models.ghostnet"])
class GhostNetV3W050(GhostNetV3Variant):
    available_weights = []

    # Parameters
    width = 0.5
    config = "default"


@kimm_export(parent_path=["kimm.models", "kimm.models.ghostnet"])
class GhostNetV3W100(GhostNetV3Variant):
    available_weights = [
        (
            "imagenet",
            GhostNetV3.default_origin,
            "ghostnetv3w100_ghostnetv3-1.0.keras",
        )
    ]

    # Parameters
    width = 1.0
    config = "default"


@kimm_export(parent_path=["kimm.models", "kimm.models.ghostnet"])
class GhostNetV3W130(GhostNetV3Variant):
    available_weights = []

    # Parameters
    width = 1.3
    config = "default"


@kimm_export(parent_path=["kimm.models", "kimm.models.ghostnet"])
class GhostNetV3W160(GhostNetV3Variant):
    available_weights = []

    # Parameters
    width = 1.6
    config = "default"


add_model_to_registry(GhostNetV3W050)
add_model_to_registry(GhostNetV3W100, "imagenet")
add_model_to_registry(GhostNetV3W130)
add_model_to_registry(GhostNetV3W160)

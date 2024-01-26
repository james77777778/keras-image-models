import math
import typing

import keras
from keras import backend
from keras import layers
from keras import ops

from kimm.blocks import apply_conv2d_block
from kimm.blocks import apply_inverted_residual_block
from kimm.blocks import apply_mlp_block
from kimm.blocks import apply_transformer_block
from kimm.models.base_model import BaseModel
from kimm.utils import add_model_to_registry
from kimm.utils import make_divisible

# type, repeat, kernel_size, channels, strides, expansion_ratio,
# transformer_dim, transformer_depth, patch_size
DEFAULT_V1_S_CONFIG = [
    ["ir", 1, 3, 32, 1, 4.0, None, None, None],
    ["ir", 3, 3, 64, 2, 4.0, None, None, None],
    ["mobilevit", 1, 3, 96, 2, 4.0, 144, 2, 2],
    ["mobilevit", 1, 3, 128, 2, 4.0, 192, 4, 2],
    ["mobilevit", 1, 3, 160, 2, 4.0, 240, 3, 2],
]
DEFAULT_V1_XS_CONFIG = [
    ["ir", 1, 3, 32, 1, 4.0, None, None, None],
    ["ir", 3, 3, 48, 2, 4.0, None, None, None],
    ["mobilevit", 1, 3, 64, 2, 4.0, 96, 2, 2],
    ["mobilevit", 1, 3, 80, 2, 4.0, 120, 4, 2],
    ["mobilevit", 1, 3, 96, 2, 4.0, 144, 3, 2],
]
DEFAULT_V1_XXS_CONFIG = [
    ["ir", 1, 3, 16, 1, 2.0, None, None, None],
    ["ir", 3, 3, 24, 2, 2.0, None, None, None],
    ["mobilevit", 1, 3, 48, 2, 2.0, 64, 2, 2],
    ["mobilevit", 1, 3, 64, 2, 2.0, 80, 4, 2],
    ["mobilevit", 1, 3, 80, 2, 2.0, 96, 3, 2],
]
DEFAULT_V2_CONFIG = [
    ["ir", 1, 3, 64, 1, 2.0, None, None, None],
    ["ir", 2, 3, 128, 2, 2.0, None, None, None],
    ["mobilevitv2", 1, 3, 256, 2, 2.0, 128, 2, 2],
    ["mobilevitv2", 1, 3, 384, 2, 2.0, 192, 4, 2],
    ["mobilevitv2", 1, 3, 512, 2, 2.0, 256, 3, 2],
]


def unfold(inputs, patch_size):
    # TODO: improve performance
    x = inputs
    h, w, c = x.shape[-3], x.shape[-2], x.shape[-1]
    new_h, new_w = (
        math.ceil(h / patch_size) * patch_size,
        math.ceil(w / patch_size) * patch_size,
    )
    num_patches_h = new_h // patch_size
    num_patches_w = new_w // patch_size
    num_patches = num_patches_h * num_patches_w
    # [B, H, W, C] -> [B * P, N, C]
    x = ops.reshape(
        x, [-1, num_patches_h, patch_size, num_patches_w, patch_size, c]
    )
    x = ops.transpose(x, [0, 2, 4, 1, 3, 5])
    x = ops.reshape(x, [-1, num_patches, c])
    return x


def fold(inputs, h, w, c, patch_size):
    # TODO: improve performance
    x = inputs
    new_h, new_w = (
        math.ceil(h / patch_size) * patch_size,
        math.ceil(w / patch_size) * patch_size,
    )
    num_patches_h = new_h // patch_size
    num_patches_w = new_w // patch_size
    # [B * P, N, C] -> [B, P, N, C] -> [B, H, W, C]
    x = ops.reshape(
        x, [-1, patch_size, patch_size, num_patches_h, num_patches_w, c]
    )
    x = ops.transpose(x, [0, 3, 1, 4, 2, 5])
    x = ops.reshape(
        x, [-1, num_patches_h * patch_size, num_patches_w * patch_size, c]
    )
    return x


def apply_mobilevit_block(
    inputs,
    output_channels: int,
    kernel_size: int = 3,
    strides: int = 1,
    expansion_ratio: float = 1.0,
    mlp_ratio: float = 2.0,
    transformer_dim: typing.Optional[int] = None,
    transformer_depth: int = 2,
    patch_size: int = 8,
    num_heads: int = 4,
    activation="swish",
    transformer_activation="swish",
    fusion: bool = True,
    name="mobilevit_block",
):
    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3
    input_channels = inputs.shape[channels_axis]
    transformer_dim = transformer_dim or make_divisible(
        input_channels * expansion_ratio
    )

    x = inputs

    # Local representation
    x = apply_conv2d_block(
        x,
        input_channels,
        kernel_size,
        strides,
        activation=activation,
        name=f"{name}_conv_kxk",
    )
    x = layers.Conv2D(
        transformer_dim, 1, use_bias=False, name=f"{name}_conv_1x1"
    )(x)

    # TODO: natively support channels_first
    # Unfold (feature map -> patches)
    if backend.image_data_format() == "channels_first":
        x = ops.transpose(x, [0, 2, 3, 1])

    h, w, c = x.shape[-3], x.shape[-2], x.shape[-1]
    x = unfold(x, patch_size)

    # Global representations
    for i in range(transformer_depth):
        x = apply_transformer_block(
            x,
            transformer_dim,
            num_heads,
            mlp_ratio,
            True,
            False,
            activation=transformer_activation,
            name=f"{name}_transformer_{i}",
        )
    x = layers.LayerNormalization(axis=-1, epsilon=1e-6, name=f"{name}_norm")(x)

    # Fold (patch -> feature map)
    x = fold(x, h, w, c, patch_size)

    # TODO: natively support channels_first
    if backend.image_data_format() == "channels_first":
        x = ops.transpose(x, [0, 3, 1, 2])

    x = apply_conv2d_block(
        x,
        output_channels,
        1,
        1,
        activation=activation,
        name=f"{name}_conv_proj",
    )
    if fusion:
        x = layers.Concatenate(axis=channels_axis)([inputs, x])

    x = apply_conv2d_block(
        x,
        output_channels,
        kernel_size,
        1,
        activation=activation,
        name=f"{name}_conv_fusion",
    )
    return x


def unfold_v2(inputs, patch_size):
    x = inputs

    if backend.image_data_format() == "channels_last":
        h, w, c = x.shape[-3], x.shape[-2], x.shape[-1]
    else:
        c, h, w = x.shape[-3], x.shape[-2], x.shape[-1]

    new_h, new_w = (
        math.ceil(h / patch_size) * patch_size,
        math.ceil(w / patch_size) * patch_size,
    )
    num_patches_h = new_h // patch_size
    num_patches_w = new_w // patch_size
    num_patches = num_patches_h * num_patches_w

    if backend.image_data_format() == "channels_last":
        # [B, H, W, C] -> [B, P, N, C]
        x = ops.reshape(
            x, [-1, num_patches_h, patch_size, num_patches_w, patch_size, c]
        )
        x = ops.transpose(x, [0, 2, 4, 1, 3, 5])
        x = ops.reshape(x, [-1, patch_size * patch_size, num_patches, c])
    else:
        # [B, C, H, W] -> [B, C, P, N]
        x = ops.reshape(
            x, [-1, c, num_patches_h, patch_size, num_patches_w, patch_size]
        )
        x = ops.transpose(x, [0, 1, 3, 5, 2, 4])
        x = ops.reshape(x, [-1, c, patch_size * patch_size, num_patches])
    return x


def fold_v2(inputs, h, w, c, patch_size):
    x = inputs

    new_h, new_w = (
        math.ceil(h / patch_size) * patch_size,
        math.ceil(w / patch_size) * patch_size,
    )
    num_patches_h = new_h // patch_size
    num_patches_w = new_w // patch_size
    if backend.image_data_format() == "channels_last":
        # [B, P, N, C] -> [B, H, W, C]
        x = ops.reshape(
            x, [-1, patch_size, patch_size, num_patches_h, num_patches_w, c]
        )
        x = ops.transpose(x, [0, 3, 1, 4, 2, 5])
        x = ops.reshape(
            x, [-1, num_patches_h * patch_size, num_patches_w * patch_size, c]
        )
    else:
        # [B, C, P, N] -> [B, C, H, W]
        x = ops.reshape(
            x, [-1, c, patch_size, patch_size, num_patches_h, num_patches_w]
        )
        x = ops.transpose(x, [0, 1, 4, 2, 5, 3])
        x = ops.reshape(
            x, [-1, c, num_patches_h * patch_size, num_patches_w * patch_size]
        )
    return x


def apply_linear_self_attention_block(
    inputs, dim: int, use_bias=True, name="linear_self_attention_block"
):
    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3
    num_patch_axis = (
        -2 if backend.image_data_format() == "channels_last" else -1
    )

    x = inputs

    # [B, P, N, C] -> [B, P, N, h + 2d]
    # Project x into query, key and value
    # Query: [B, P, N, 1]
    # Value & Key: [B, P, N, d]
    x = layers.Conv2D(
        1 + (2 * dim), 1, use_bias=use_bias, name=f"{name}_qkv_proj_conv2d"
    )(x)
    query, key, value = ops.split(x, [1, 1 + dim], axis=channels_axis)

    # Apply softmax along N dimension
    context_scores = ops.softmax(query, axis=num_patch_axis)

    # Compute context vector
    # [B, P, N, d] x [B, P, N, 1] -> [B, P, N, d] -> [B, P, 1, d]
    context_vector = layers.Multiply()([key, context_scores])
    context_vector = ops.sum(context_vector, axis=num_patch_axis, keepdims=True)

    # Combine context vector with values
    # [B, P, N, d] * [B, P, 1, d] -> [B, P, N, d]
    out = layers.ReLU()(value)
    out = layers.Multiply()([out, context_vector])
    out = layers.Conv2D(
        dim, 1, use_bias=use_bias, name=f"{name}_out_proj_conv2d"
    )(out)
    return out


def apply_linear_transformer_block(
    inputs,
    dim: int,
    mlp_ratio: float = 2.0,
    activation="swish",
    name="linear_transformer_block",
):
    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3
    x = inputs

    # Self-attention
    x = layers.GroupNormalization(
        1, axis=channels_axis, epsilon=1e-5, name=f"{name}_norm1"
    )(x)
    x = apply_linear_self_attention_block(
        x, dim, use_bias=True, name=f"{name}_attn"
    )
    x = layers.Add()([inputs, x])

    # Feedforward network
    residual = x
    x = layers.GroupNormalization(
        1, axis=channels_axis, epsilon=1e-5, name=f"{name}_norm2"
    )(x)
    x = apply_mlp_block(
        x,
        int(dim * mlp_ratio),
        activation=activation,
        use_bias=True,
        use_conv_mlp=True,
        name=f"{name}_mlp",
    )
    x = layers.Add()([residual, x])
    return x


def apply_mobilevitv2_block(
    inputs,
    output_channels: int,
    kernel_size: int = 3,
    strides: int = 1,
    expansion_ratio: float = 1.0,
    mlp_ratio: float = 2.0,
    transformer_dim: typing.Optional[int] = None,
    transformer_depth: int = 2,
    patch_size: int = 8,
    activation="swish",
    transformer_activation="swish",
    name="mobilevitv2_block",
):
    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3
    input_channels = inputs.shape[channels_axis]
    transformer_dim = transformer_dim or make_divisible(
        input_channels * expansion_ratio
    )

    x = inputs

    # Local representation
    x = apply_conv2d_block(
        x,
        input_channels,
        kernel_size,
        strides,
        activation=activation,
        use_depthwise=True,
        name=f"{name}_conv_kxk",
    )
    x = layers.Conv2D(
        transformer_dim, 1, use_bias=False, name=f"{name}_conv_1x1"
    )(x)

    # Unfold (feature map -> patches)
    if backend.image_data_format() == "channels_last":
        h, w, c = x.shape[-3], x.shape[-2], x.shape[-1]
    else:
        c, h, w = x.shape[-3], x.shape[-2], x.shape[-1]
    x = unfold_v2(x, patch_size)

    # Global representations:
    for i in range(transformer_depth):
        x = apply_linear_transformer_block(
            x,
            transformer_dim,
            mlp_ratio,
            activation=transformer_activation,
            name=f"{name}_transformer_{i}",
        )
    x = layers.GroupNormalization(
        1, axis=channels_axis, epsilon=1e-5, name=f"{name}_norm"
    )(x)

    # Fold (patch -> feature map)
    x = fold_v2(x, h, w, c, patch_size)

    x = apply_conv2d_block(
        x,
        output_channels,
        1,
        1,
        activation=None,
        name=f"{name}_conv_proj",
    )
    return x


@keras.saving.register_keras_serializable(package="kimm")
class MobileViT(BaseModel):
    available_feature_keys = [
        "STEM_S2",
        *[f"BLOCK{i}_S{j}" for i, j in zip(range(5), [2, 4, 8, 16, 32])],
    ]

    def __init__(
        self,
        stem_channels: int = 16,
        head_channels: int = 640,
        activation: str = "swish",
        config: str = "v1_s",
        **kwargs,
    ):
        kwargs["weights_url"] = self.get_weights_url(kwargs["weights"])

        _available_configs = ["v1_s", "v1_xs", "v1_xss"]
        if config == "v1_s":
            _config = DEFAULT_V1_S_CONFIG
        elif config == "v1_xs":
            _config = DEFAULT_V1_XS_CONFIG
        elif config == "v1_xxs":
            _config = DEFAULT_V1_XXS_CONFIG
        else:
            raise ValueError(
                f"config must be one of {_available_configs} using string. "
                f"Received: config={config}"
            )

        input_tensor = kwargs.pop("input_tensor", None)
        self.set_properties(kwargs, 256)

        inputs = self.determine_input_tensor(
            input_tensor,
            self._input_shape,
            self._default_size,
            static_shape=True,
        )
        x = inputs

        x = self.build_preprocessing(x, "0_1")

        # Prepare feature extraction
        features = {}

        # stem
        x = apply_conv2d_block(
            x, stem_channels, 3, 2, activation=activation, name="stem"
        )
        features["STEM_S2"] = x

        # blocks
        current_stride = 2
        for current_block_idx, cfg in enumerate(_config):
            (
                block_type,
                r,
                k,
                c,
                s,
                e,
                transformer_dim,
                transformer_depth,
                patch_size,
            ) = cfg
            # always apply inverted_residual_block
            for current_layer_idx in range(r):
                s = s if current_layer_idx == 0 else 1
                name = f"stages_{current_block_idx}_{current_layer_idx}"
                x = apply_inverted_residual_block(
                    x, c, k, 1, 1, s, e, activation=activation, name=name
                )
                current_stride *= s
            if block_type == "mobilevit":
                name = f"stages_{current_block_idx}_{current_layer_idx + 1}"
                x = apply_mobilevit_block(
                    x,
                    c,
                    k,
                    1,
                    transformer_dim=transformer_dim,
                    transformer_depth=transformer_depth,
                    patch_size=patch_size,
                    activation=activation,
                    transformer_activation=activation,
                    name=name,
                )
            features[f"BLOCK{current_block_idx}_S{current_stride}"] = x

        # last conv block
        x = apply_conv2d_block(
            x, head_channels, 1, 1, activation=activation, name="final_conv"
        )

        # Head
        x = self.build_head(x)

        super().__init__(inputs=inputs, outputs=x, features=features, **kwargs)

        # All references to `self` below this line
        self.stem_channels = stem_channels
        self.head_channels = head_channels
        self.activation = activation
        self.config = config

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stem_channels": self.stem_channels,
                "head_channels": self.head_channels,
                "activation": self.activation,
                "config": self.config,
            }
        )
        return config

    def fix_config(self, config):
        unused_kwargs = [
            "stem_channels",
            "head_channels",
            "activation",
            "config",
        ]
        for k in unused_kwargs:
            config.pop(k, None)
        return config


@keras.saving.register_keras_serializable(package="kimm")
class MobileViTV2(BaseModel):
    available_feature_keys = [
        "STEM_S2",
        *[f"BLOCK{i}_S{j}" for i, j in zip(range(5), [2, 4, 8, 16, 32])],
    ]

    def __init__(
        self,
        multiplier: float = 1.0,
        activation: str = "swish",
        config: str = "v2",
        **kwargs,
    ):
        kwargs["weights_url"] = self.get_weights_url(kwargs["weights"])

        _available_configs = ["v2"]
        if config == "v2":
            _config = DEFAULT_V2_CONFIG
        else:
            raise ValueError(
                f"config must be one of {_available_configs} using string. "
                f"Received: config={config}"
            )

        input_tensor = kwargs.pop("input_tensor", None)
        self.set_properties(kwargs, 256)

        inputs = self.determine_input_tensor(
            input_tensor,
            self._input_shape,
            self._default_size,
            static_shape=True,
        )
        x = inputs

        x = self.build_preprocessing(x, "0_1")

        # Prepare feature extraction
        features = {}

        # stem
        x = apply_conv2d_block(
            x,
            int(32 * multiplier),
            3,
            2,
            activation=activation,
            name="stem",
        )
        features["STEM_S2"] = x

        # blocks
        current_stride = 2
        for current_block_idx, cfg in enumerate(_config):
            (
                block_type,
                r,
                k,
                c,
                s,
                e,
                _,
                transformer_depth,
                patch_size,
            ) = cfg
            c = int(c * multiplier)
            # always apply inverted_residual_block
            for current_layer_idx in range(r):
                s = s if current_layer_idx == 0 else 1
                name = f"stages_{current_block_idx}_{current_layer_idx}"
                x = apply_inverted_residual_block(
                    x, c, k, 1, 1, s, e, activation=activation, name=name
                )
                current_stride *= s
            if block_type == "mobilevitv2":
                name = f"stages_{current_block_idx}_{current_layer_idx + 1}"
                x = apply_mobilevitv2_block(
                    x,
                    c,
                    k,
                    1,
                    0.5,
                    mlp_ratio=2.0,
                    transformer_depth=transformer_depth,
                    patch_size=patch_size,
                    activation=activation,
                    transformer_activation=activation,
                    name=name,
                )
            features[f"BLOCK{current_block_idx}_S{current_stride}"] = x

        # Head
        x = self.build_head(x)

        super().__init__(inputs=inputs, outputs=x, features=features, **kwargs)

        # All references to `self` below this line
        self.multiplier = multiplier
        self.activation = activation
        self.config = config

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "multiplier": self.multiplier,
                "activation": self.activation,
                "config": self.config,
            }
        )
        return config

    def fix_config(self, config):
        unused_kwargs = ["multiplier", "activation", "config"]
        for k in unused_kwargs:
            config.pop(k, None)
        return config


class MobileViTXXS(MobileViT):
    available_weights = [
        (
            "imagenet",
            MobileViT.default_origin,
            "mobilevitxxs_mobilevit_xxs.cvnets_in1k.keras",
        )
    ]

    def __init__(
        self,
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        dropout_rate: float = 0.1,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = "imagenet",
        config: str = "v1_xxs",
        name="MobileViTXXS",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            16,
            320,
            "swish",
            config,
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


class MobileViTXS(MobileViT):
    available_weights = [
        (
            "imagenet",
            MobileViT.default_origin,
            "mobilevitxs_mobilevit_xs.cvnets_in1k.keras",
        )
    ]

    def __init__(
        self,
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        dropout_rate: float = 0.1,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = "imagenet",
        config: str = "v1_xs",
        name="MobileViTXS",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            16,
            384,
            "swish",
            config,
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


class MobileViTS(MobileViT):
    available_weights = [
        (
            "imagenet",
            MobileViT.default_origin,
            "mobilevits_mobilevit_s.cvnets_in1k.keras",
        )
    ]

    def __init__(
        self,
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        dropout_rate: float = 0.1,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = "imagenet",
        config: str = "v1_s",
        name="MobileViTS",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            16,
            640,
            "swish",
            config,
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


class MobileViTV2W050(MobileViTV2):
    available_weights = [
        (
            "imagenet",
            MobileViTV2.default_origin,
            "mobilevitv2w050_mobilevitv2_050.cvnets_in1k.keras",
        )
    ]

    def __init__(
        self,
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        dropout_rate: float = 0.1,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = "imagenet",
        config: str = "v2",
        name="MobileViTV2W050",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            0.5,
            "swish",
            config,
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


class MobileViTV2W075(MobileViTV2):
    available_weights = [
        (
            "imagenet",
            MobileViTV2.default_origin,
            "mobilevitv2w075_mobilevitv2_075.cvnets_in1k.keras",
        )
    ]

    def __init__(
        self,
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        dropout_rate: float = 0.1,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = "imagenet",
        config: str = "v2",
        name="MobileViTV2W075",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            0.75,
            "swish",
            config,
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


class MobileViTV2W100(MobileViTV2):
    available_weights = [
        (
            "imagenet",
            MobileViTV2.default_origin,
            "mobilevitv2w100_mobilevitv2_100.cvnets_in1k.keras",
        )
    ]

    def __init__(
        self,
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        dropout_rate: float = 0.1,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = "imagenet",
        config: str = "v2",
        name="MobileViTV2W100",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            1.0,
            "swish",
            config,
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


class MobileViTV2W125(MobileViTV2):
    available_weights = [
        (
            "imagenet",
            MobileViTV2.default_origin,
            "mobilevitv2w125_mobilevitv2_125.cvnets_in1k.keras",
        )
    ]

    def __init__(
        self,
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        dropout_rate: float = 0.1,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = "imagenet",
        config: str = "v2",
        name="MobileViTV2W125",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            1.25,
            "swish",
            config,
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


class MobileViTV2W150(MobileViTV2):
    available_weights = [
        (
            "imagenet",
            MobileViTV2.default_origin,
            "mobilevitv2w150_mobilevitv2_150.cvnets_in22k_ft_in1k_384.keras",
        )
    ]

    def __init__(
        self,
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        dropout_rate: float = 0.1,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = "imagenet",
        config: str = "v2",
        name="MobileViTV2W150",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            1.5,
            "swish",
            config,
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


class MobileViTV2W175(MobileViTV2):
    available_weights = [
        (
            "imagenet",
            MobileViTV2.default_origin,
            "mobilevitv2w175_mobilevitv2_175.cvnets_in22k_ft_in1k_384.keras",
        )
    ]

    def __init__(
        self,
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        dropout_rate: float = 0.1,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = "imagenet",
        config: str = "v2",
        name="MobileViTV2W175",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            1.75,
            "swish",
            config,
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


class MobileViTV2W200(MobileViTV2):
    available_weights = [
        (
            "imagenet",
            MobileViTV2.default_origin,
            "mobilevitv2w200_mobilevitv2_200.cvnets_in22k_ft_in1k_384.keras",
        )
    ]

    def __init__(
        self,
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        dropout_rate: float = 0.1,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = "imagenet",
        config: str = "v2",
        name="MobileViTV2W200",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            2.0,
            "swish",
            config,
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


add_model_to_registry(MobileViTXXS, "imagenet")
add_model_to_registry(MobileViTXS, "imagenet")
add_model_to_registry(MobileViTS, "imagenet")
add_model_to_registry(MobileViTV2W050, "imagenet")
add_model_to_registry(MobileViTV2W075, "imagenet")
add_model_to_registry(MobileViTV2W100, "imagenet")
add_model_to_registry(MobileViTV2W125, "imagenet")
add_model_to_registry(MobileViTV2W150, "imagenet")
add_model_to_registry(MobileViTV2W175, "imagenet")
add_model_to_registry(MobileViTV2W200, "imagenet")

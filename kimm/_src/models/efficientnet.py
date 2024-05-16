import math
import typing

import keras
from keras import backend
from keras import layers

from kimm._src.blocks.conv2d import apply_conv2d_block
from kimm._src.blocks.depthwise_separation import (
    apply_depthwise_separation_block,
)
from kimm._src.blocks.inverted_residual import apply_inverted_residual_block
from kimm._src.kimm_export import kimm_export
from kimm._src.models.base_model import BaseModel
from kimm._src.utils.make_divisble import make_divisible
from kimm._src.utils.model_registry import add_model_to_registry

# type, repeat, kernel_size, strides, expansion_ratio, channels, se_ratio
# ds: depthwise separation block
# ir: inverted residual block
# er: edge residual block
# cn: normal conv block
DEFAULT_V1_CONFIG = [
    ["ds", 1, 3, 1, 1, 16, 0.25],
    ["ir", 2, 3, 2, 6, 24, 0.25],
    ["ir", 2, 5, 2, 6, 40, 0.25],
    ["ir", 3, 3, 2, 6, 80, 0.25],
    ["ir", 3, 5, 1, 6, 112, 0.25],
    ["ir", 4, 5, 2, 6, 192, 0.25],
    ["ir", 1, 3, 1, 6, 320, 0.25],
]
DEFAULT_V1_LITE_CONFIG = [
    ["ds", 1, 3, 1, 1, 16, 0.0],
    ["ir", 2, 3, 2, 6, 24, 0.0],
    ["ir", 2, 5, 2, 6, 40, 0.0],
    ["ir", 3, 3, 2, 6, 80, 0.0],
    ["ir", 3, 5, 1, 6, 112, 0.0],
    ["ir", 4, 5, 2, 6, 192, 0.0],
    ["ir", 1, 3, 1, 6, 320, 0.0],
]
DEFAULT_V2_S_CONFIG = [
    ["cn", 2, 3, 1, 1, 24, 0.0],
    ["er", 4, 3, 2, 4, 48, 0.0],
    ["er", 4, 3, 2, 4, 64, 0.0],
    ["ir", 6, 3, 2, 4, 128, 0.25],
    ["ir", 9, 3, 1, 6, 160, 0.25],
    ["ir", 15, 3, 2, 6, 256, 0.25],
]
DEFAULT_V2_M_CONFIG = [
    ["cn", 3, 3, 1, 1, 24, 0.0],
    ["er", 5, 3, 2, 4, 48, 0.0],
    ["er", 5, 3, 2, 4, 80, 0.0],
    ["ir", 7, 3, 2, 4, 160, 0.25],
    ["ir", 14, 3, 1, 6, 176, 0.25],
    ["ir", 18, 3, 2, 6, 304, 0.25],
    ["ir", 5, 3, 1, 6, 512, 0.25],
]
DEFAULT_V2_L_CONFIG = [
    ["cn", 4, 3, 1, 1, 32, 0.0],
    ["er", 7, 3, 2, 4, 64, 0.0],
    ["er", 7, 3, 2, 4, 96, 0.0],
    ["ir", 10, 3, 2, 4, 192, 0.25],
    ["ir", 19, 3, 1, 6, 224, 0.25],
    ["ir", 25, 3, 2, 6, 384, 0.25],
    ["ir", 7, 3, 1, 6, 640, 0.25],
]
DEFAULT_V2_XL_CONFIG = [
    ["cn", 4, 3, 1, 1, 32, 0.0],
    ["er", 8, 3, 2, 4, 64, 0.0],
    ["er", 8, 3, 2, 4, 96, 0.0],
    ["ir", 16, 3, 2, 4, 192, 0.25],
    ["ir", 24, 3, 1, 6, 256, 0.25],
    ["ir", 32, 3, 2, 6, 512, 0.25],
    ["ir", 8, 3, 1, 6, 640, 0.25],
]
DEFAULT_V2_BASE_CONFIG = [
    ["cn", 1, 3, 1, 1, 16, 0.0],
    ["er", 2, 3, 2, 4, 32, 0.0],
    ["er", 2, 3, 2, 4, 48, 0.0],
    ["ir", 3, 3, 2, 4, 96, 0.25],
    ["ir", 5, 3, 1, 6, 112, 0.25],
    ["ir", 8, 3, 2, 6, 192, 0.25],
]


def apply_edge_residual_block(
    inputs,
    output_channels,
    expansion_kernel_size=1,
    pointwise_kernel_size=1,
    strides=1,
    expansion_ratio=1.0,
    activation="swish",
    bn_epsilon=1e-5,
    padding=None,
    name="edge_residual_block",
):
    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3
    input_channels = inputs.shape[channels_axis]
    hidden_channels = make_divisible(input_channels * expansion_ratio)
    has_skip = strides == 1 and input_channels == output_channels

    x = inputs
    # Expansion
    x = apply_conv2d_block(
        x,
        hidden_channels,
        expansion_kernel_size,
        strides,
        activation=activation,
        bn_epsilon=bn_epsilon,
        padding=padding,
        name=f"{name}_conv_exp",
    )
    # Point-wise linear projection
    x = apply_conv2d_block(
        x,
        output_channels,
        pointwise_kernel_size,
        1,
        activation=None,
        bn_epsilon=bn_epsilon,
        padding=padding,
        name=f"{name}_conv_pwl",
    )
    if has_skip:
        x = layers.Add()([x, inputs])
    return x


@keras.saving.register_keras_serializable(package="kimm")
class EfficientNet(BaseModel):
    # for: v1, v1_lite, v2_m, v2_l, v2_xl, tinynet
    # not for: v2_s, v2_base
    available_feature_keys = [
        "STEM_S2",
        *[
            f"BLOCK{i}_S{j}"
            for i, j in zip(range(7), [2, 4, 8, 16, 16, 32, 32])
        ],
    ]

    def __init__(
        self,
        width: float = 1.0,
        depth: float = 1.0,
        stem_channels: int = 32,
        head_channels: int = 1280,
        fix_stem_and_head_channels: bool = False,
        fix_first_and_last_blocks: bool = False,
        activation="swish",
        config: str = "v1",
        **kwargs,
    ):
        kwargs["weights_url"] = self.get_weights_url(kwargs["weights"])

        _available_configs = [
            "v1",
            "v1_lite",
            "v2_s",
            "v2_m",
            "v2_l",
            "v2_xl",
            "v2_base",
        ]
        if config == "v1":
            _config = DEFAULT_V1_CONFIG
        elif config == "v1_lite":
            _config = DEFAULT_V1_LITE_CONFIG
        elif config == "v2_s":
            _config = DEFAULT_V2_S_CONFIG
        elif config == "v2_m":
            _config = DEFAULT_V2_M_CONFIG
        elif config == "v2_l":
            _config = DEFAULT_V2_L_CONFIG
        elif config == "v2_xl":
            _config = DEFAULT_V2_XL_CONFIG
        elif config == "v2_base":
            _config = DEFAULT_V2_BASE_CONFIG
        else:
            raise ValueError(
                f"config must be one of {_available_configs} using string. "
                f"Received: config={config}"
            )
        # TF default config
        bn_epsilon = kwargs.pop("bn_epsilon", 1e-5)
        padding = kwargs.pop("padding", None)
        # EfficientNetV2Base config
        round_limit = kwargs.pop("round_limit", 0.9)
        # TinyNet config
        round_fn = kwargs.pop("round_fn", math.ceil)

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
        stem_channel = (
            stem_channels
            if fix_stem_and_head_channels
            else make_divisible(stem_channels * width)
        )
        x = apply_conv2d_block(
            x,
            stem_channel,
            3,
            2,
            activation=activation,
            bn_epsilon=bn_epsilon,
            padding=padding,
            name="conv_stem",
        )
        features["STEM_S2"] = x

        # Blocks
        current_stride = 2
        for current_block_idx, cfg in enumerate(_config):
            block_type, r, k, s, e, c, se = cfg
            c = make_divisible(c * width, round_limit=round_limit)
            if fix_first_and_last_blocks and (
                current_block_idx in (0, len(_config) - 1)
            ):
                r = r
            else:
                r = int(round_fn(r * depth))
            for current_layer_idx in range(r):
                s = s if current_layer_idx == 0 else 1
                _kwargs = {
                    "bn_epsilon": bn_epsilon,
                    "padding": padding,
                    "name": f"blocks_{current_block_idx}_{current_layer_idx}",
                    "activation": activation,
                }
                if block_type == "ds":
                    x = apply_depthwise_separation_block(
                        x, c, k, 1, s, se, se_activation=activation, **_kwargs
                    )
                elif block_type == "ir":
                    se_c = x.shape[channels_axis]
                    x = apply_inverted_residual_block(
                        x, c, k, 1, 1, s, e, se, se_channels=se_c, **_kwargs
                    )
                elif block_type == "cn":
                    x = apply_conv2d_block(x, c, k, s, add_skip=True, **_kwargs)
                elif block_type == "er":
                    x = apply_edge_residual_block(x, c, k, 1, s, e, **_kwargs)
                current_stride *= s
            features[f"BLOCK{current_block_idx}_S{current_stride}"] = x

        # Last conv block
        if fix_stem_and_head_channels:
            conv_head_channels = head_channels
        else:
            conv_head_channels = make_divisible(head_channels * width)
        x = apply_conv2d_block(
            x,
            conv_head_channels,
            1,
            1,
            activation=activation,
            bn_epsilon=bn_epsilon,
            name="conv_head",
        )

        # Head
        x = self.build_head(x)

        super().__init__(inputs=inputs, outputs=x, features=features, **kwargs)

        # All references to `self` below this line
        self.width = width
        self.depth = depth
        self.stem_channels = stem_channels
        self.head_channels = head_channels
        self.fix_stem_and_head_channels = fix_stem_and_head_channels
        self.fix_first_and_last_blocks = fix_first_and_last_blocks
        self.activation = activation
        self.config = config

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "width": self.width,
                "depth": self.depth,
                "stem_channels": self.stem_channels,
                "head_channels": self.head_channels,
                "fix_stem_and_head_channels": self.fix_stem_and_head_channels,
                "fix_first_and_last_blocks": self.fix_first_and_last_blocks,
                "activation": self.activation,
                "config": self.config,
            }
        )
        return config

    def fix_config(self, config: typing.Dict):
        unused_kwargs = [
            "width",
            "depth",
            "stem_channels",
            "head_channels",
            "fix_stem_and_head_channels",
            "fix_first_and_last_blocks",
            "activation",
            "config",
        ]
        for k in unused_kwargs:
            config.pop(k, None)
        return config


# Model Definition


class EfficientNetVariant(EfficientNet):
    # Parameters
    width = None
    depth = None
    stem_channels = None
    head_channels = None
    fix_stem_and_head_channels = None
    fix_first_and_last_blocks = None
    activation = None
    config = None
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
        if type(self) is EfficientNetVariant:
            raise NotImplementedError(
                f"Cannot instantiate base class: {self.__class__.__name__}. "
                "You should use its subclasses."
            )
        kwargs = self.fix_config(kwargs)
        if hasattr(self, "round_limit"):
            kwargs["round_limit"] = self.round_limit
        if hasattr(self, "bn_epsilon"):
            kwargs["bn_epsilon"] = self.bn_epsilon
        if hasattr(self, "padding"):
            kwargs["padding"] = self.padding
        if hasattr(self, "round_fn"):
            kwargs["round_fn"] = self.round_fn
        # default to TF configuration (bn_epsilon=1e-3 and padding="same")
        super().__init__(
            width=self.width,
            depth=self.depth,
            stem_channels=self.stem_channels,
            head_channels=self.head_channels,
            fix_stem_and_head_channels=self.fix_stem_and_head_channels,
            fix_first_and_last_blocks=self.fix_first_and_last_blocks,
            activation=self.activation,
            config=self.config,
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


@kimm_export(parent_path=["kimm.models", "kimm.models.efficientnet"])
class EfficientNetB0(EfficientNetVariant):
    available_weights = [
        (
            "imagenet",
            EfficientNet.default_origin,
            "efficientnetb0_tf_efficientnet_b0.ns_jft_in1k.keras",
        )
    ]

    # Parameters
    width = 1.0
    depth = 1.0
    stem_channels = 32
    head_channels = 1280
    fix_stem_and_head_channels = False
    fix_first_and_last_blocks = False
    activation = "swish"
    config = "v1"
    default_size = 224
    bn_epsilon = 1e-3
    padding = "same"


@kimm_export(parent_path=["kimm.models", "kimm.models.efficientnet"])
class EfficientNetB1(EfficientNetVariant):
    available_weights = [
        (
            "imagenet",
            EfficientNet.default_origin,
            "efficientnetb1_tf_efficientnet_b1.ns_jft_in1k.keras",
        )
    ]

    # Parameters
    width = 1.0
    depth = 1.1
    stem_channels = 32
    head_channels = 1280
    fix_stem_and_head_channels = False
    fix_first_and_last_blocks = False
    activation = "swish"
    config = "v1"
    default_size = 240
    bn_epsilon = 1e-3
    padding = "same"


@kimm_export(parent_path=["kimm.models", "kimm.models.efficientnet"])
class EfficientNetB2(EfficientNetVariant):
    available_weights = [
        (
            "imagenet",
            EfficientNet.default_origin,
            "efficientnetb2_tf_efficientnet_b2.ns_jft_in1k.keras",
        )
    ]

    # Parameters
    width = 1.1
    depth = 1.2
    stem_channels = 32
    head_channels = 1280
    fix_stem_and_head_channels = False
    fix_first_and_last_blocks = False
    activation = "swish"
    config = "v1"
    default_size = 260
    bn_epsilon = 1e-3
    padding = "same"


@kimm_export(parent_path=["kimm.models", "kimm.models.efficientnet"])
class EfficientNetB3(EfficientNetVariant):
    available_weights = [
        (
            "imagenet",
            EfficientNet.default_origin,
            "efficientnetb3_tf_efficientnet_b3.ns_jft_in1k.keras",
        )
    ]

    # Parameters
    width = 1.2
    depth = 1.4
    stem_channels = 32
    head_channels = 1280
    fix_stem_and_head_channels = False
    fix_first_and_last_blocks = False
    activation = "swish"
    config = "v1"
    default_size = 300
    bn_epsilon = 1e-3
    padding = "same"


@kimm_export(parent_path=["kimm.models", "kimm.models.efficientnet"])
class EfficientNetB4(EfficientNetVariant):
    available_weights = [
        (
            "imagenet",
            EfficientNet.default_origin,
            "efficientnetb4_tf_efficientnet_b4.ns_jft_in1k.keras",
        )
    ]

    # Parameters
    width = 1.4
    depth = 1.8
    stem_channels = 32
    head_channels = 1280
    fix_stem_and_head_channels = False
    fix_first_and_last_blocks = False
    activation = "swish"
    config = "v1"
    default_size = 380
    bn_epsilon = 1e-3
    padding = "same"


@kimm_export(parent_path=["kimm.models", "kimm.models.efficientnet"])
class EfficientNetB5(EfficientNetVariant):
    available_weights = [
        (
            "imagenet",
            EfficientNet.default_origin,
            "efficientnetb5_tf_efficientnet_b5.ns_jft_in1k.keras",
        )
    ]

    # Parameters
    width = 1.6
    depth = 2.2
    stem_channels = 32
    head_channels = 1280
    fix_stem_and_head_channels = False
    fix_first_and_last_blocks = False
    activation = "swish"
    config = "v1"
    default_size = 456
    bn_epsilon = 1e-3
    padding = "same"


@kimm_export(parent_path=["kimm.models", "kimm.models.efficientnet"])
class EfficientNetB6(EfficientNetVariant):
    available_weights = [
        (
            "imagenet",
            EfficientNet.default_origin,
            "efficientnetb6_tf_efficientnet_b6.ns_jft_in1k.keras",
        )
    ]

    # Parameters
    width = 1.8
    depth = 2.6
    stem_channels = 32
    head_channels = 1280
    fix_stem_and_head_channels = False
    fix_first_and_last_blocks = False
    activation = "swish"
    config = "v1"
    default_size = 528
    bn_epsilon = 1e-3
    padding = "same"


@kimm_export(parent_path=["kimm.models", "kimm.models.efficientnet"])
class EfficientNetB7(EfficientNetVariant):
    available_weights = [
        (
            "imagenet",
            EfficientNet.default_origin,
            "efficientnetb7_tf_efficientnet_b7.ns_jft_in1k.keras",
        )
    ]

    # Parameters
    width = 2.0
    depth = 3.1
    stem_channels = 32
    head_channels = 1280
    fix_stem_and_head_channels = False
    fix_first_and_last_blocks = False
    activation = "swish"
    config = "v1"
    default_size = 600
    bn_epsilon = 1e-3
    padding = "same"


@kimm_export(parent_path=["kimm.models", "kimm.models.efficientnet"])
class EfficientNetLiteB0(EfficientNetVariant):
    available_weights = [
        (
            "imagenet",
            EfficientNet.default_origin,
            "efficientnetliteb0_tf_efficientnet_lite0.in1k.keras",
        )
    ]

    # Parameters
    width = 1.0
    depth = 1.0
    stem_channels = 32
    head_channels = 1280
    fix_stem_and_head_channels = True
    fix_first_and_last_blocks = True
    activation = "relu6"
    config = "v1_lite"
    default_size = 224
    bn_epsilon = 1e-3
    padding = "same"


@kimm_export(parent_path=["kimm.models", "kimm.models.efficientnet"])
class EfficientNetLiteB1(EfficientNetVariant):
    available_weights = [
        (
            "imagenet",
            EfficientNet.default_origin,
            "efficientnetliteb1_tf_efficientnet_lite1.in1k.keras",
        )
    ]

    # Parameters
    width = 1.0
    depth = 1.1
    stem_channels = 32
    head_channels = 1280
    fix_stem_and_head_channels = True
    fix_first_and_last_blocks = True
    activation = "relu6"
    config = "v1_lite"
    default_size = 240
    bn_epsilon = 1e-3
    padding = "same"


@kimm_export(parent_path=["kimm.models", "kimm.models.efficientnet"])
class EfficientNetLiteB2(EfficientNetVariant):
    available_weights = [
        (
            "imagenet",
            EfficientNet.default_origin,
            "efficientnetliteb2_tf_efficientnet_lite2.in1k.keras",
        )
    ]

    # Parameters
    width = 1.1
    depth = 1.2
    stem_channels = 32
    head_channels = 1280
    fix_stem_and_head_channels = True
    fix_first_and_last_blocks = True
    activation = "relu6"
    config = "v1_lite"
    default_size = 260
    bn_epsilon = 1e-3
    padding = "same"


@kimm_export(parent_path=["kimm.models", "kimm.models.efficientnet"])
class EfficientNetLiteB3(EfficientNetVariant):
    available_weights = [
        (
            "imagenet",
            EfficientNet.default_origin,
            "efficientnetliteb3_tf_efficientnet_lite3.in1k.keras",
        )
    ]

    # Parameters
    width = 1.2
    depth = 1.4
    stem_channels = 32
    head_channels = 1280
    fix_stem_and_head_channels = True
    fix_first_and_last_blocks = True
    activation = "relu6"
    config = "v1_lite"
    default_size = 300
    bn_epsilon = 1e-3
    padding = "same"


@kimm_export(parent_path=["kimm.models", "kimm.models.efficientnet"])
class EfficientNetLiteB4(EfficientNetVariant):
    available_weights = [
        (
            "imagenet",
            EfficientNet.default_origin,
            "efficientnetliteb4_tf_efficientnet_lite4.in1k.keras",
        )
    ]

    # Parameters
    width = 1.4
    depth = 1.8
    stem_channels = 32
    head_channels = 1280
    fix_stem_and_head_channels = True
    fix_first_and_last_blocks = True
    activation = "relu6"
    config = "v1_lite"
    default_size = 380
    bn_epsilon = 1e-3
    padding = "same"


@kimm_export(parent_path=["kimm.models", "kimm.models.efficientnet"])
class EfficientNetV2S(EfficientNetVariant):
    available_feature_keys = [
        "STEM_S2",
        *[f"BLOCK{i}_S{j}" for i, j in zip(range(6), [2, 4, 8, 16, 16, 32])],
    ]
    available_weights = [
        (
            "imagenet",
            EfficientNet.default_origin,
            "efficientnetv2s_tf_efficientnetv2_s.in21k_ft_in1k.keras",
        )
    ]

    # Parameters
    width = 1.0
    depth = 1.0
    stem_channels = 24
    head_channels = 1280
    fix_stem_and_head_channels = False
    fix_first_and_last_blocks = False
    activation = "swish"
    config = "v2_s"
    default_size = 300
    bn_epsilon = 1e-3
    padding = "same"


@kimm_export(parent_path=["kimm.models", "kimm.models.efficientnet"])
class EfficientNetV2M(EfficientNetVariant):
    available_weights = [
        (
            "imagenet",
            EfficientNet.default_origin,
            "efficientnetv2m_tf_efficientnetv2_m.in21k_ft_in1k.keras",
        )
    ]

    # Parameters
    width = 1.0
    depth = 1.0
    stem_channels = 24
    head_channels = 1280
    fix_stem_and_head_channels = False
    fix_first_and_last_blocks = False
    activation = "swish"
    config = "v2_m"
    default_size = 384
    bn_epsilon = 1e-3
    padding = "same"


@kimm_export(parent_path=["kimm.models", "kimm.models.efficientnet"])
class EfficientNetV2L(EfficientNetVariant):
    available_weights = [
        (
            "imagenet",
            EfficientNet.default_origin,
            "efficientnetv2l_tf_efficientnetv2_l.in21k_ft_in1k.keras",
        )
    ]

    # Parameters
    width = 1.0
    depth = 1.0
    stem_channels = 32
    head_channels = 1280
    fix_stem_and_head_channels = False
    fix_first_and_last_blocks = False
    activation = "swish"
    config = "v2_l"
    default_size = 384
    bn_epsilon = 1e-3
    padding = "same"


@kimm_export(parent_path=["kimm.models", "kimm.models.efficientnet"])
class EfficientNetV2XL(EfficientNetVariant):
    available_weights = [
        (
            "imagenet",
            EfficientNet.default_origin,
            "efficientnetv2xl_tf_efficientnetv2_xl.in21k_ft_in1k.keras",
        )
    ]

    # Parameters
    width = 1.0
    depth = 1.0
    stem_channels = 32
    head_channels = 1280
    fix_stem_and_head_channels = False
    fix_first_and_last_blocks = False
    activation = "swish"
    config = "v2_xl"
    default_size = 384
    bn_epsilon = 1e-3
    padding = "same"


@kimm_export(parent_path=["kimm.models", "kimm.models.efficientnet"])
class EfficientNetV2B0(EfficientNetVariant):
    available_feature_keys = [
        "STEM_S2",
        *[f"BLOCK{i}_S{j}" for i, j in zip(range(6), [2, 4, 8, 16, 16, 32])],
    ]
    available_weights = [
        (
            "imagenet",
            EfficientNet.default_origin,
            "efficientnetv2b0_tf_efficientnetv2_b0.in1k.keras",
        )
    ]

    # Parameters
    width = 1.0
    depth = 1.0
    stem_channels = 32
    head_channels = 1280
    fix_stem_and_head_channels = True
    fix_first_and_last_blocks = False
    activation = "swish"
    config = "v2_base"
    default_size = 192
    bn_epsilon = 1e-3
    padding = "same"


@kimm_export(parent_path=["kimm.models", "kimm.models.efficientnet"])
class EfficientNetV2B1(EfficientNetVariant):
    available_feature_keys = [
        "STEM_S2",
        *[f"BLOCK{i}_S{j}" for i, j in zip(range(6), [2, 4, 8, 16, 16, 32])],
    ]
    available_weights = [
        (
            "imagenet",
            EfficientNet.default_origin,
            "efficientnetv2b1_tf_efficientnetv2_b1.in1k.keras",
        )
    ]

    # Parameters
    width = 1.0
    depth = 1.1
    stem_channels = 32
    head_channels = 1280
    fix_stem_and_head_channels = True
    fix_first_and_last_blocks = False
    activation = "swish"
    config = "v2_base"
    default_size = 192
    bn_epsilon = 1e-3
    padding = "same"


@kimm_export(parent_path=["kimm.models", "kimm.models.efficientnet"])
class EfficientNetV2B2(EfficientNetVariant):
    available_feature_keys = [
        "STEM_S2",
        *[f"BLOCK{i}_S{j}" for i, j in zip(range(6), [2, 4, 8, 16, 16, 32])],
    ]
    available_weights = [
        (
            "imagenet",
            EfficientNet.default_origin,
            "efficientnetv2b2_tf_efficientnetv2_b2.in1k.keras",
        )
    ]

    # Parameters
    width = 1.1
    depth = 1.2
    stem_channels = make_divisible(32 * 1.1)
    head_channels = make_divisible(1280 * 1.1)
    fix_stem_and_head_channels = True
    fix_first_and_last_blocks = False
    activation = "swish"
    config = "v2_base"
    default_size = 208
    bn_epsilon = 1e-3
    padding = "same"
    round_limit = 0.0  # fix


@kimm_export(parent_path=["kimm.models", "kimm.models.efficientnet"])
class EfficientNetV2B3(EfficientNetVariant):
    available_feature_keys = [
        "STEM_S2",
        *[f"BLOCK{i}_S{j}" for i, j in zip(range(6), [2, 4, 8, 16, 16, 32])],
    ]
    available_weights = [
        (
            "imagenet",
            EfficientNet.default_origin,
            "efficientnetv2b3_tf_efficientnetv2_b3.in1k.keras",
        )
    ]

    # Parameters
    width = 1.2
    depth = 1.4
    stem_channels = make_divisible(32 * 1.2)
    head_channels = make_divisible(1280 * 1.2)
    fix_stem_and_head_channels = True
    fix_first_and_last_blocks = False
    activation = "swish"
    config = "v2_base"
    default_size = 240
    bn_epsilon = 1e-3
    padding = "same"
    round_limit = 0.0  # fix


@kimm_export(parent_path=["kimm.models", "kimm.models.efficientnet"])
class TinyNetA(EfficientNetVariant):
    available_weights = [
        (
            "imagenet",
            EfficientNet.default_origin,
            "tinyneta_tinynet_a.in1k.keras",
        )
    ]

    # Parameters
    width = 1.0
    depth = 1.2
    stem_channels = 32
    head_channels = 1280
    fix_stem_and_head_channels = False
    fix_first_and_last_blocks = False
    activation = "swish"
    config = "v1"
    default_size = 192
    round_fn = round  # tinynet config


@kimm_export(parent_path=["kimm.models", "kimm.models.efficientnet"])
class TinyNetB(EfficientNetVariant):
    available_weights = [
        (
            "imagenet",
            EfficientNet.default_origin,
            "tinynetb_tinynet_b.in1k.keras",
        )
    ]

    # Parameters
    width = 0.75
    depth = 1.1
    stem_channels = 32
    head_channels = 1280
    fix_stem_and_head_channels = True
    fix_first_and_last_blocks = False
    activation = "swish"
    config = "v1"
    default_size = 192
    round_fn = round  # tinynet config


@kimm_export(parent_path=["kimm.models", "kimm.models.efficientnet"])
class TinyNetC(EfficientNetVariant):
    available_weights = [
        (
            "imagenet",
            EfficientNet.default_origin,
            "tinynetc_tinynet_c.in1k.keras",
        )
    ]

    # Parameters
    width = 0.54
    depth = 0.85
    stem_channels = 32
    head_channels = 1280
    fix_stem_and_head_channels = True
    fix_first_and_last_blocks = False
    activation = "swish"
    config = "v1"
    default_size = 188
    round_fn = round  # tinynet config


@kimm_export(parent_path=["kimm.models", "kimm.models.efficientnet"])
class TinyNetD(EfficientNetVariant):
    available_weights = [
        (
            "imagenet",
            EfficientNet.default_origin,
            "tinynetd_tinynet_d.in1k.keras",
        )
    ]

    # Parameters
    width = 0.54
    depth = 0.695
    stem_channels = 32
    head_channels = 1280
    fix_stem_and_head_channels = True
    fix_first_and_last_blocks = False
    activation = "swish"
    config = "v1"
    default_size = 152
    round_fn = round  # tinynet config


@kimm_export(parent_path=["kimm.models", "kimm.models.efficientnet"])
class TinyNetE(EfficientNetVariant):
    available_weights = [
        (
            "imagenet",
            EfficientNet.default_origin,
            "tinynete_tinynet_e.in1k.keras",
        )
    ]

    # Parameters
    width = 0.51
    depth = 0.6
    stem_channels = 32
    head_channels = 1280
    fix_stem_and_head_channels = True
    fix_first_and_last_blocks = False
    activation = "swish"
    config = "v1"
    default_size = 106
    round_fn = round  # tinynet config


add_model_to_registry(EfficientNetB0, "imagenet")
add_model_to_registry(EfficientNetB1, "imagenet")
add_model_to_registry(EfficientNetB2, "imagenet")
add_model_to_registry(EfficientNetB3, "imagenet")
add_model_to_registry(EfficientNetB4, "imagenet")
add_model_to_registry(EfficientNetB5, "imagenet")
add_model_to_registry(EfficientNetB6, "imagenet")
add_model_to_registry(EfficientNetB7, "imagenet")
add_model_to_registry(EfficientNetLiteB0, "imagenet")
add_model_to_registry(EfficientNetLiteB1, "imagenet")
add_model_to_registry(EfficientNetLiteB2, "imagenet")
add_model_to_registry(EfficientNetLiteB3, "imagenet")
add_model_to_registry(EfficientNetLiteB4, "imagenet")
add_model_to_registry(EfficientNetV2S, "imagenet")
add_model_to_registry(EfficientNetV2M, "imagenet")
add_model_to_registry(EfficientNetV2L, "imagenet")
add_model_to_registry(EfficientNetV2XL, "imagenet")
add_model_to_registry(EfficientNetV2B0, "imagenet")
add_model_to_registry(EfficientNetV2B1, "imagenet")
add_model_to_registry(EfficientNetV2B2, "imagenet")
add_model_to_registry(EfficientNetV2B3, "imagenet")
add_model_to_registry(TinyNetA, "imagenet")
add_model_to_registry(TinyNetB, "imagenet")
add_model_to_registry(TinyNetC, "imagenet")
add_model_to_registry(TinyNetD, "imagenet")
add_model_to_registry(TinyNetE, "imagenet")

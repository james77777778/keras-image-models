import math
import typing

import keras
from keras import layers
from keras import utils

from kimm.blocks import apply_conv2d_block
from kimm.blocks import apply_depthwise_separation_block
from kimm.blocks import apply_inverted_residual_block
from kimm.models.base_model import BaseModel
from kimm.utils import add_model_to_registry
from kimm.utils import make_divisible

DEFAULT_SMALL_CONFIG = [
    # type, repeat, kernel_size, strides, expansion_ratio, channels, se_ratio,
    # activation
    # stage0
    [["ds", 1, 3, 2, 1.0, 16, 0.25, "relu"]],
    # stage1
    [
        ["ir", 1, 3, 2, 4.5, 24, 0.0, "relu"],
        ["ir", 1, 3, 1, 3.67, 24, 0.0, "relu"],
    ],
    # stage2
    [
        ["ir", 1, 5, 2, 4.0, 40, 0.25, "hard_swish"],
        ["ir", 2, 5, 1, 6.0, 40, 0.25, "hard_swish"],
    ],
    # stage3
    [["ir", 2, 5, 1, 3.0, 48, 0.25, "hard_swish"]],
    # stage4
    [["ir", 3, 5, 2, 6.0, 96, 0.25, "hard_swish"]],
    # stage5
    [["cn", 1, 1, 1, 1.0, 576, 0.0, "hard_swish"]],
]
DEFAULT_LARGE_CONFIG = [
    # type, repeat, kernel_size, strides, expansion_ratio, channels, se_ratio,
    # activation
    # stage0
    [["ds", 1, 3, 1, 1.0, 16, 0.0, "relu"]],
    # stage1
    [
        ["ir", 1, 3, 2, 4.0, 24, 0.0, "relu"],
        ["ir", 1, 3, 1, 3.0, 24, 0.0, "relu"],
    ],
    # stage2
    [["ir", 3, 5, 2, 3.0, 40, 0.25, "relu"]],
    # stage3
    [
        ["ir", 1, 3, 2, 6.0, 80, 0.0, "hard_swish"],
        ["ir", 1, 3, 1, 2.5, 80, 0.0, "hard_swish"],
        ["ir", 2, 3, 1, 2.3, 80, 0.0, "hard_swish"],
    ],
    # stage4
    [["ir", 2, 3, 1, 6.0, 112, 0.25, "hard_swish"]],
    # stage5
    [["ir", 3, 5, 2, 6.0, 160, 0.25, "hard_swish"]],
    # stage6
    [["cn", 1, 1, 1, 1.0, 960, 0.0, "hard_swish"]],
]
DEFAULT_LCNET_CONFIG = [
    # type, repeat, kernel_size, strides, expansion_ratio, channels, se_ratio,
    # activation
    # stage0
    [["dsa", 1, 3, 1, 1.0, 32, 0.0, "hard_swish"]],
    # stage1
    [["dsa", 2, 3, 2, 1.0, 64, 0.0, "hard_swish"]],
    # stage2
    [["dsa", 2, 3, 2, 1.0, 128, 0.0, "hard_swish"]],
    # stage3
    [
        ["dsa", 1, 3, 2, 1.0, 256, 0.0, "hard_swish"],
        ["dsa", 1, 5, 1, 1.0, 256, 0.0, "hard_swish"],
    ],
    # stage4
    [["dsa", 4, 5, 1, 1.0, 256, 0.0, "hard_swish"]],
    # stage5
    [["dsa", 2, 5, 2, 1.0, 512, 0.25, "hard_swish"]],
]


class MobileNetV3(BaseModel):
    def __init__(
        self,
        width: float = 1.0,
        depth: float = 1.0,
        fix_stem_and_head_channels: bool = False,
        config: typing.Union[str, typing.List] = "large",
        minimal: bool = False,
        **kwargs,
    ):
        _available_configs = ["small", "large", "lcnet"]
        if config == "small":
            _config = DEFAULT_SMALL_CONFIG
            conv_head_channels = 1024
        elif config == "large":
            _config = DEFAULT_LARGE_CONFIG
            conv_head_channels = 1280
        elif config == "lcnet":
            _config = DEFAULT_LCNET_CONFIG
            conv_head_channels = 1280
        else:
            raise ValueError(
                f"config must be one of {_available_configs} using string. "
                f"Received: config={config}"
            )
        if minimal:
            force_activation = "relu"
            force_kernel_size = 3
            no_se = True
        else:
            force_activation = None
            force_kernel_size = None
            no_se = False
        # TF default config
        bn_epsilon = kwargs.pop("bn_epsilon", 1e-5)
        padding = kwargs.pop("padding", None)

        parsed_kwargs = self.parse_kwargs(kwargs)
        img_input = self.determine_input_tensor(
            parsed_kwargs["input_tensor"],
            parsed_kwargs["input_shape"],
            parsed_kwargs["default_size"],
        )
        x = img_input

        if parsed_kwargs["include_preprocessing"]:
            x = self.build_preprocessing(x, "imagenet")

        # Prepare feature extraction
        features = {}

        # stem
        stem_channel = (
            16 if fix_stem_and_head_channels else make_divisible(16 * width)
        )
        x = apply_conv2d_block(
            x,
            stem_channel,
            3,
            2,
            activation=force_activation or "hard_swish",
            bn_epsilon=bn_epsilon,
            padding=padding,
            name="conv_stem",
        )
        features["STEM_S2"] = x

        # blocks
        current_stride = 2
        for current_stage_idx, cfg in enumerate(_config):
            for current_block_idx, sub_cfg in enumerate(cfg):
                block_type, r, k, s, e, c, se, act = sub_cfg

                # override default config
                if force_activation is not None:
                    act = force_activation
                if force_kernel_size is not None:
                    k = force_kernel_size if k > force_kernel_size else k
                if no_se:
                    se = 0.0

                c = make_divisible(c * width)
                # no depth multiplier at first and last block
                if current_block_idx not in (0, len(_config) - 1):
                    r = int(math.ceil(r * depth))
                for current_layer_idx in range(r):
                    s = s if current_layer_idx == 0 else 1
                    _kwargs = {
                        "bn_epsilon": bn_epsilon,
                        "padding": padding,
                        "name": (
                            f"blocks_{current_stage_idx}_"
                            f"{current_block_idx + current_layer_idx}"
                        ),
                    }
                    if block_type in ("ds", "dsa"):
                        x = apply_depthwise_separation_block(
                            x,
                            c,
                            k,
                            1,
                            s,
                            se,
                            act,
                            se_activation="relu",
                            se_gate_activation="hard_sigmoid",
                            se_make_divisible_number=8,
                            pw_activation=act if block_type == "dsa" else None,
                            skip=False if block_type == "dsa" else True,
                            **_kwargs,
                        )
                    elif block_type == "ir":
                        x = apply_inverted_residual_block(
                            x,
                            c,
                            k,
                            1,
                            1,
                            s,
                            e,
                            se,
                            act,
                            se_activation="relu",
                            se_gate_activation="hard_sigmoid",
                            se_make_divisible_number=8,
                            **_kwargs,
                        )
                    elif block_type == "cn":
                        x = apply_conv2d_block(
                            x, c, k, s, activation=act, **_kwargs
                        )
                    current_stride *= s
            features[f"BLOCK{current_stage_idx}_S{current_stride}"] = x

        # Head
        if parsed_kwargs["include_top"]:
            if fix_stem_and_head_channels:
                conv_head_channels = conv_head_channels
            else:
                conv_head_channels = max(
                    conv_head_channels,
                    make_divisible(conv_head_channels * width),
                )
            head_activation = force_activation or "hard_swish"
            x = self.build_top(
                x,
                parsed_kwargs["classes"],
                parsed_kwargs["classifier_activation"],
                parsed_kwargs["dropout_rate"],
                conv_head_channels=conv_head_channels,
                head_activation=head_activation,
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
        self.width = width
        self.depth = depth
        self.fix_stem_and_head_channels = fix_stem_and_head_channels
        self.config = config
        self.minimal = minimal

    def build_top(
        self,
        inputs,
        classes,
        classifier_activation,
        dropout_rate,
        conv_head_channels,
        head_activation,
    ):
        x = layers.GlobalAveragePooling2D(name="avg_pool", keepdims=True)(
            inputs
        )
        x = layers.Conv2D(
            conv_head_channels, 1, 1, use_bias=True, name="conv_head"
        )(x)
        x = layers.Activation(head_activation, name="act2")(x)
        x = layers.Flatten()(x)
        x = layers.Dropout(rate=dropout_rate, name="conv_head_dropout")(x)
        x = layers.Dense(
            classes, activation=classifier_activation, name="classifier"
        )(x)
        return x

    @staticmethod
    def available_feature_keys():
        raise NotImplementedError()

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "width": self.width,
                "depth": self.depth,
                "fix_stem_and_head_channels": self.fix_stem_and_head_channels,
                "config": self.config,
                "minimal": self.minimal,
            }
        )
        return config

    def fix_config(self, config):
        unused_kwargs = [
            "width",
            "depth",
            "fix_stem_and_head_channels",
            "config",
            "minimal",
        ]
        for k in unused_kwargs:
            config.pop(k, None)
        return config


"""
Model Definition
"""


class MobileNet050V3Small(MobileNetV3):
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
        config: typing.Union[str, typing.List] = "small",
        name: str = "MobileNet050V3Small",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            0.5,
            1.0,
            True,
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

    @staticmethod
    def available_feature_keys():
        feature_keys = ["STEM_S2"]
        feature_keys.extend(
            [f"BLOCK{i}_S{j}" for i, j in zip(range(6), [4, 8, 16, 16, 32, 32])]
        )
        return feature_keys


class MobileNet075V3Small(MobileNetV3):
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
        config: typing.Union[str, typing.List] = "small",
        name: str = "MobileNet075V3Small",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            0.75,
            1.0,
            False,
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

    @staticmethod
    def available_feature_keys():
        feature_keys = ["STEM_S2"]
        feature_keys.extend(
            [f"BLOCK{i}_S{j}" for i, j in zip(range(6), [4, 8, 16, 16, 32, 32])]
        )
        return feature_keys


class MobileNet100V3Small(MobileNetV3):
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
        config: typing.Union[str, typing.List] = "small",
        name: str = "MobileNet100V3Small",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            1.0,
            1.0,
            False,
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

    @staticmethod
    def available_feature_keys():
        feature_keys = ["STEM_S2"]
        feature_keys.extend(
            [f"BLOCK{i}_S{j}" for i, j in zip(range(6), [4, 8, 16, 16, 32, 32])]
        )
        return feature_keys


class MobileNet100V3SmallMinimal(MobileNetV3):
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
        config: typing.Union[str, typing.List] = "small",
        name: str = "MobileNet100V3SmallMinimal",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        # default to TF configuration (bn_epsilon=1e-3 and padding="same")
        super().__init__(
            1.0,
            1.0,
            False,
            config,
            True,
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
            bn_epsilon=1e-3,
            padding="same",
            **kwargs,
        )

    @staticmethod
    def available_feature_keys():
        feature_keys = ["STEM_S2"]
        feature_keys.extend(
            [f"BLOCK{i}_S{j}" for i, j in zip(range(6), [4, 8, 16, 16, 32, 32])]
        )
        return feature_keys


class MobileNet100V3Large(MobileNetV3):
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
        config: typing.Union[str, typing.List] = "large",
        name: str = "MobileNet100V3Large",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            1.0,
            1.0,
            False,
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

    @staticmethod
    def available_feature_keys():
        feature_keys = ["STEM_S2"]
        feature_keys.extend(
            [
                f"BLOCK{i}_S{j}"
                for i, j in zip(range(7), [2, 4, 8, 16, 16, 32, 32])
            ]
        )
        return feature_keys


class MobileNet100V3LargeMinimal(MobileNetV3):
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
        config: typing.Union[str, typing.List] = "large",
        name: str = "MobileNet100V3LargeMinimal",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        # default to TF configuration (bn_epsilon=1e-3 and padding="same")
        super().__init__(
            1.0,
            1.0,
            False,
            config,
            True,
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
            bn_epsilon=1e-3,
            padding="same",
            **kwargs,
        )

    @staticmethod
    def available_feature_keys():
        feature_keys = ["STEM_S2"]
        feature_keys.extend(
            [
                f"BLOCK{i}_S{j}"
                for i, j in zip(range(7), [2, 4, 8, 16, 16, 32, 32])
            ]
        )
        return feature_keys


class LCNet035(MobileNetV3):
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
        config: typing.Union[str, typing.List] = "lcnet",
        name: str = "LCNet035",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        # default to TF configuration (bn_epsilon=1e-3 and padding="same")
        super().__init__(
            0.35,
            1.0,
            False,
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

    @staticmethod
    def available_feature_keys():
        feature_keys = ["STEM_S2"]
        feature_keys.extend(
            [f"BLOCK{i}_S{j}" for i, j in zip(range(6), [2, 4, 8, 16, 16, 32])]
        )
        return feature_keys


class LCNet050(MobileNetV3):
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
        config: typing.Union[str, typing.List] = "lcnet",
        name: str = "LCNet050",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        # default to TF configuration (bn_epsilon=1e-3 and padding="same")
        super().__init__(
            0.5,
            1.0,
            False,
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

    @staticmethod
    def available_feature_keys():
        feature_keys = ["STEM_S2"]
        feature_keys.extend(
            [f"BLOCK{i}_S{j}" for i, j in zip(range(6), [2, 4, 8, 16, 16, 32])]
        )
        return feature_keys


class LCNet075(MobileNetV3):
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
        config: typing.Union[str, typing.List] = "lcnet",
        name: str = "LCNet075",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        # default to TF configuration (bn_epsilon=1e-3 and padding="same")
        super().__init__(
            0.75,
            1.0,
            False,
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

    @staticmethod
    def available_feature_keys():
        feature_keys = ["STEM_S2"]
        feature_keys.extend(
            [f"BLOCK{i}_S{j}" for i, j in zip(range(6), [2, 4, 8, 16, 16, 32])]
        )
        return feature_keys


class LCNet100(MobileNetV3):
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
        config: typing.Union[str, typing.List] = "lcnet",
        name: str = "LCNet100",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        # default to TF configuration (bn_epsilon=1e-3 and padding="same")
        super().__init__(
            1.0,
            1.0,
            False,
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

    @staticmethod
    def available_feature_keys():
        feature_keys = ["STEM_S2"]
        feature_keys.extend(
            [f"BLOCK{i}_S{j}" for i, j in zip(range(6), [2, 4, 8, 16, 16, 32])]
        )
        return feature_keys


class LCNet150(MobileNetV3):
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
        config: typing.Union[str, typing.List] = "lcnet",
        name: str = "LCNet150",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        # default to TF configuration (bn_epsilon=1e-3 and padding="same")
        super().__init__(
            1.5,
            1.0,
            False,
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

    @staticmethod
    def available_feature_keys():
        feature_keys = ["STEM_S2"]
        feature_keys.extend(
            [
                f"BLOCK{i}_S{j}"
                for i, j in zip(range(7), [2, 4, 8, 16, 16, 32, 32])
            ]
        )
        return feature_keys


add_model_to_registry(MobileNet050V3Small, "imagenet")
add_model_to_registry(MobileNet075V3Small, "imagenet")
add_model_to_registry(MobileNet100V3Small, "imagenet")
add_model_to_registry(MobileNet100V3SmallMinimal, "imagenet")
add_model_to_registry(MobileNet100V3Large, "imagenet")
add_model_to_registry(MobileNet100V3LargeMinimal, "imagenet")
add_model_to_registry(LCNet035)
add_model_to_registry(LCNet050, "imagenet")
add_model_to_registry(LCNet075, "imagenet")
add_model_to_registry(LCNet100, "imagenet")
add_model_to_registry(LCNet150)

import math
import typing

import keras

from kimm._src.blocks.conv2d import apply_conv2d_block
from kimm._src.blocks.depthwise_separation import (
    apply_depthwise_separation_block,
)
from kimm._src.blocks.inverted_residual import apply_inverted_residual_block
from kimm._src.kimm_export import kimm_export
from kimm._src.models.base_model import BaseModel
from kimm._src.utils.make_divisble import make_divisible
from kimm._src.utils.model_registry import add_model_to_registry

DEFAULT_CONFIG = [
    # type, repeat, kernel_size, strides, expansion_ratio, channels
    ["ds", 1, 3, 1, 1, 16],
    ["ir", 2, 3, 2, 6, 24],
    ["ir", 3, 3, 2, 6, 32],
    ["ir", 4, 3, 2, 6, 64],
    ["ir", 3, 3, 1, 6, 96],
    ["ir", 3, 3, 2, 6, 160],
    ["ir", 1, 3, 1, 6, 320],
]


@keras.saving.register_keras_serializable(package="kimm")
class MobileNetV2(BaseModel):
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
        fix_stem_and_head_channels: bool = False,
        config: typing.Literal["default"] = "default",
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
        )
        x = inputs

        x = self.build_preprocessing(x, "imagenet")

        # Prepare feature extraction
        features = {}

        # stem
        stem_channel = (
            32 if fix_stem_and_head_channels else make_divisible(32 * width)
        )
        x = apply_conv2d_block(
            x,
            stem_channel,
            3,
            2,
            activation="relu6",
            name="conv_stem",
        )
        features["STEM_S2"] = x

        # blocks
        current_stride = 2
        for current_block_idx, cfg in enumerate(_config):
            block_type, r, k, s, e, c = cfg
            c = make_divisible(c * width)
            # no depth multiplier at first and last block
            if current_block_idx not in (0, len(_config) - 1):
                r = int(math.ceil(r * depth))
            for current_layer_idx in range(r):
                s = s if current_layer_idx == 0 else 1
                name = f"blocks_{current_block_idx}_{current_layer_idx}"
                if block_type == "ds":
                    x = apply_depthwise_separation_block(
                        x, c, k, 1, s, activation="relu6", name=name
                    )
                elif block_type == "ir":
                    x = apply_inverted_residual_block(
                        x, c, k, 1, 1, s, e, activation="relu6", name=name
                    )
                current_stride *= s
            features[f"BLOCK{current_block_idx}_S{current_stride}"] = x

        # last conv
        if fix_stem_and_head_channels:
            head_channels = 1280
        else:
            head_channels = max(1280, make_divisible(1280 * width))
        x = apply_conv2d_block(
            x, head_channels, 1, 1, activation="relu6", name="conv_head"
        )

        # Head
        x = self.build_head(x)

        super().__init__(inputs=inputs, outputs=x, features=features, **kwargs)

        # All references to `self` below this line
        self.width = width
        self.depth = depth
        self.fix_stem_and_head_channels = fix_stem_and_head_channels
        self.config = config

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "width": self.width,
                "depth": self.depth,
                "fix_stem_and_head_channels": self.fix_stem_and_head_channels,
                "config": self.config,
            }
        )
        return config

    def fix_config(self, config):
        unused_kwargs = [
            "width",
            "depth",
            "fix_stem_and_head_channels",
            "config",
        ]
        for k in unused_kwargs:
            config.pop(k, None)
        return config


# Model Definition


class MobileNetV2Variant(MobileNetV2):
    # Parameters
    width = None
    depth = None
    fix_stem_and_head_channels = None
    config = None

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
        if type(self) is MobileNetV2Variant:
            raise NotImplementedError(
                f"Cannot instantiate base class: {self.__class__.__name__}. "
                "You should use its subclasses."
            )
        kwargs = self.fix_config(kwargs)
        super().__init__(
            width=self.width,
            depth=self.depth,
            fix_stem_and_head_channels=self.fix_stem_and_head_channels,
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
            **kwargs,
        )


@kimm_export(parent_path=["kimm.models", "kimm.models.mobilenet_v2"])
class MobileNetV2W050(MobileNetV2Variant):
    available_weights = [
        (
            "imagenet",
            MobileNetV2.default_origin,
            "mobilenet050v2_mobilenetv2_050.lamb_in1k.keras",
        )
    ]

    # Parameters
    width = 0.5
    depth = 1.0
    fix_stem_and_head_channels = False
    config = "default"


@kimm_export(parent_path=["kimm.models", "kimm.models.mobilenet_v2"])
class MobileNetV2W100(MobileNetV2Variant):
    available_weights = [
        (
            "imagenet",
            MobileNetV2.default_origin,
            "mobilenet100v2_mobilenetv2_100.ra_in1k.keras",
        )
    ]

    # Parameters
    width = 1.0
    depth = 1.0
    fix_stem_and_head_channels = False
    config = "default"


@kimm_export(parent_path=["kimm.models", "kimm.models.mobilenet_v2"])
class MobileNetV2W110(MobileNetV2Variant):
    available_weights = [
        (
            "imagenet",
            MobileNetV2.default_origin,
            "mobilenet110v2_mobilenetv2_110d.ra_in1k.keras",
        )
    ]

    # Parameters
    width = 1.1
    depth = 1.2
    fix_stem_and_head_channels = False
    config = "default"


@kimm_export(parent_path=["kimm.models", "kimm.models.mobilenet_v2"])
class MobileNetV2W120(MobileNetV2Variant):
    available_weights = [
        (
            "imagenet",
            MobileNetV2.default_origin,
            "mobilenet120v2_mobilenetv2_120d.ra_in1k.keras",
        )
    ]

    # Parameters
    width = 1.2
    depth = 1.4
    fix_stem_and_head_channels = False
    config = "default"


@kimm_export(parent_path=["kimm.models", "kimm.models.mobilenet_v2"])
class MobileNetV2W140(MobileNetV2Variant):
    available_weights = [
        (
            "imagenet",
            MobileNetV2.default_origin,
            "mobilenet140v2_mobilenetv2_140.ra_in1k.keras",
        )
    ]

    # Parameters
    width = 1.4
    depth = 1.0
    fix_stem_and_head_channels = False
    config = "default"


add_model_to_registry(MobileNetV2W050, "imagenet")
add_model_to_registry(MobileNetV2W100, "imagenet")
add_model_to_registry(MobileNetV2W110, "imagenet")
add_model_to_registry(MobileNetV2W120, "imagenet")
add_model_to_registry(MobileNetV2W140, "imagenet")

import math
import pathlib
import typing

import keras
from keras import backend

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
                    has_skip = x.shape[channels_axis] == c and s == 1
                    x = apply_depthwise_separation_block(
                        x,
                        c,
                        k,
                        1,
                        s,
                        activation="relu6",
                        has_skip=has_skip,
                        name=name,
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
        """Instantiates the MobileNetV2 architecture.

        Reference:
        - [MobileNetV2: Inverted Residuals and Linear Bottlenecks (CVPR 2018)]
        (https://arxiv.org/abs/1801.04381)

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
            feature_extractor=feature_extractor,
            feature_keys=feature_keys,
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

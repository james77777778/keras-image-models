import typing

import keras
import numpy as np
from keras import backend
from keras import layers

from kimm.blocks import apply_conv2d_block
from kimm.blocks import apply_se_block
from kimm.models.base_model import BaseModel
from kimm.utils import add_model_to_registry


def _adjust_widths_and_groups(widths, groups, expansion_ratio):
    def _quantize_float(f, q):
        return int(round(f / q) * q)

    bottleneck_widths = [int(w * b) for w, b in zip(widths, expansion_ratio)]
    groups = [min(g, w_bot) for g, w_bot in zip(groups, bottleneck_widths)]
    bottleneck_widths = [
        _quantize_float(w_bot, g) for w_bot, g in zip(bottleneck_widths, groups)
    ]
    widths = [
        int(w_bot / b) for w_bot, b in zip(bottleneck_widths, expansion_ratio)
    ]
    return widths, groups


def _generate_regnet(
    width_init,
    width_slope,
    width_mult,
    group_size,
    depth,
    quant=8,
    expansion_ratio=1.0,
):
    widths_cont = np.arange(depth) * width_slope + width_init
    width_exps = np.round(np.log(widths_cont / width_init) / np.log(width_mult))
    widths = (
        np.round(
            np.divide(width_init * np.power(width_mult, width_exps), quant)
        )
        * quant
    )
    num_stages = len(np.unique(widths))
    groups = np.array([group_size for _ in range(num_stages)])

    widths = np.array(widths).astype(int).tolist()
    stage_gs = groups.astype(int).tolist()

    # Convert to per-stage format
    stage_widths, stage_depths = np.unique(widths, return_counts=True)
    stage_e = [expansion_ratio for _ in range(num_stages)]
    stage_strides = []
    for _ in range(num_stages):
        stride = 2
        stage_strides.append(stride)

    # Adjust the compatibility of ws and gws
    stage_widths, stage_gs = _adjust_widths_and_groups(
        stage_widths, stage_gs, stage_e
    )
    per_stage_args = [
        params
        for params in zip(
            stage_widths,
            stage_strides,
            stage_depths,
            stage_e,
            stage_gs,
        )
    ]
    return per_stage_args


def apply_bottleneck_block(
    inputs,
    output_channels: int,
    strides: int = 1,
    expansion_ratio: float = 1.0,
    group_size: int = 1,
    se_ratio: float = 0.25,
    activation="relu",
    linear_out: bool = False,
    name="bottleneck_block",
):
    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3
    input_channels = inputs.shape[channels_axis]
    expansion_channels = int(round(output_channels * expansion_ratio))
    groups = expansion_channels // group_size

    shortcut = inputs
    x = inputs
    x = apply_conv2d_block(
        x,
        expansion_channels,
        1,
        1,
        activation=activation,
        name=f"{name}_conv1",
    )
    x = apply_conv2d_block(
        x,
        expansion_channels,
        3,
        strides,
        groups=groups,
        activation=activation,
        name=f"{name}_conv2",
    )
    if se_ratio > 0.0:
        x = apply_se_block(
            x,
            se_ratio,
            activation,
            se_input_channels=input_channels,
            name=f"{name}_se",
        )
    x = apply_conv2d_block(
        x,
        output_channels,
        1,
        1,
        activation=None,
        name=f"{name}_conv3",
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
    if not linear_out:
        x = layers.Activation(activation=activation, name=f"{name}")(x)
    return x


@keras.saving.register_keras_serializable(package="kimm")
class RegNet(BaseModel):
    available_feature_keys = [
        "STEM_S2",
        *[f"BLOCK{i}_S{j}" for i, j in zip(range(4), [4, 8, 16, 32])],
    ]

    def __init__(
        self,
        w0: int = 80,
        wa: float = 42.64,
        wm: float = 2.66,
        group_size: int = 24,
        depth: int = 21,
        se_ratio: float = 0.0,
        **kwargs,
    ):
        kwargs["weights_url"] = self.get_weights_url(kwargs["weights"])

        per_stage_config = _generate_regnet(w0, wa, wm, group_size, depth)

        input_tensor = kwargs.pop("input_tensor", None)
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
        stem_channels = 32
        x = apply_conv2d_block(
            x, stem_channels, 3, 2, activation="relu", name="stem"
        )
        features["STEM_S2"] = x

        # stages
        current_stride = 2
        for current_stage_idx, params in enumerate(per_stage_config):
            c, s, d, e, g = params
            current_stride *= s
            # blocks
            for current_block_idx in range(d):
                s = s if current_block_idx == 0 else 1
                name = f"s{current_stage_idx + 1}_b{current_block_idx + 1}"
                x = apply_bottleneck_block(x, c, s, e, g, se_ratio, name=name)
            # add feature
            features[f"BLOCK{current_stage_idx}_S{current_stride}"] = x

        # Head
        x = self.build_head(x)

        super().__init__(inputs=inputs, outputs=x, features=features, **kwargs)

        # All references to `self` below this line
        self.w0 = w0
        self.wa = wa
        self.wm = wm
        self.group_size = group_size
        self.depth = depth
        self.se_ratio = se_ratio

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "w0": self.w0,
                "wa": self.wa,
                "wm": self.wm,
                "group_size": self.group_size,
                "depth": self.depth,
                "se_ratio": self.se_ratio,
            }
        )
        return config

    def fix_config(self, config):
        unused_kwargs = ["w0", "wa", "wm", "group_size", "depth", "se_ratio"]
        for k in unused_kwargs:
            config.pop(k, None)
        return config


"""
Model Definition
"""


class RegNetX002(RegNet):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnetx002_regnetx_002.pycls_in1k.keras",
        )
    ]

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
        name: str = "RegNetX002",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            24,
            36.44,
            2.49,
            8,
            13,
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


class RegNetY002(RegNet):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnety002_regnety_002.pycls_in1k.keras",
        )
    ]

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
        name: str = "RegNetY002",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            24,
            36.44,
            2.49,
            8,
            13,
            0.25,
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


class RegNetX004(RegNet):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnetx004_regnetx_004.pycls_in1k.keras",
        )
    ]

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
        name: str = "RegNetX004",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            24,
            24.48,
            2.54,
            16,
            22,
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


class RegNetY004(RegNet):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnety004_regnety_004.tv2_in1k.keras",
        )
    ]

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
        name: str = "RegNetY004",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            48,
            27.89,
            2.09,
            8,
            16,
            0.25,
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


class RegNetX006(RegNet):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnetx006_regnetx_006.pycls_in1k.keras",
        )
    ]

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
        name: str = "RegNetX006",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            48,
            36.97,
            2.24,
            24,
            16,
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


class RegNetY006(RegNet):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnety006_regnety_006.pycls_in1k.keras",
        )
    ]

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
        name: str = "RegNetY006",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            48,
            32.54,
            2.32,
            16,
            15,
            0.25,
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


class RegNetX008(RegNet):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnetx008_regnetx_008.tv2_in1k.keras",
        )
    ]

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
        name: str = "RegNetX008",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            56,
            35.73,
            2.28,
            16,
            16,
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


class RegNetY008(RegNet):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnety008_regnety_008.pycls_in1k.keras",
        )
    ]

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
        name: str = "RegNetY008",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            56,
            38.84,
            2.4,
            16,
            14,
            0.25,
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


class RegNetX016(RegNet):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnetx016_regnetx_016.tv2_in1k.keras",
        )
    ]

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
        name: str = "RegNetX016",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            80,
            34.01,
            2.25,
            24,
            18,
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


class RegNetY016(RegNet):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnety016_regnety_016.tv2_in1k.keras",
        )
    ]

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
        name: str = "RegNetY016",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            48,
            20.71,
            2.65,
            24,
            27,
            0.25,
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


class RegNetX032(RegNet):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnetx032_regnetx_032.tv2_in1k.keras",
        )
    ]

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
        name: str = "RegNetX032",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            88,
            26.31,
            2.25,
            48,
            25,
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


class RegNetY032(RegNet):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnety032_regnety_032.ra_in1k.keras",
        )
    ]

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
        name: str = "RegNetY032",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            80,
            42.63,
            2.66,
            24,
            21,
            0.25,
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


class RegNetX040(RegNet):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnetx040_regnetx_040.pycls_in1k.keras",
        )
    ]

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
        name: str = "RegNetX040",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            96,
            38.65,
            2.43,
            40,
            23,
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


class RegNetY040(RegNet):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnety040_regnety_040.ra3_in1k.keras",
        )
    ]

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
        name: str = "RegNetY040",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            96,
            31.41,
            2.24,
            64,
            22,
            0.25,
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


class RegNetX064(RegNet):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnetx064_regnetx_064.pycls_in1k.keras",
        )
    ]

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
        name: str = "RegNetX064",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            184,
            60.83,
            2.07,
            56,
            17,
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


class RegNetY064(RegNet):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnety064_regnety_064.ra3_in1k.keras",
        )
    ]

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
        name: str = "RegNetY064",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            112,
            33.22,
            2.27,
            72,
            25,
            0.25,
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


class RegNetX080(RegNet):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnetx080_regnetx_080.tv2_in1k.keras",
        )
    ]

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
        name: str = "RegNetX080",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            80,
            49.56,
            2.88,
            120,
            23,
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


class RegNetY080(RegNet):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnety080_regnety_080.ra3_in1k.keras",
        )
    ]

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
        name: str = "RegNetY080",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            192,
            76.82,
            2.19,
            56,
            17,
            0.25,
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


class RegNetX120(RegNet):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnetx120_regnetx_120.pycls_in1k.keras",
        )
    ]

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
        name: str = "RegNetX120",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            168,
            73.36,
            2.37,
            112,
            19,
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


class RegNetY120(RegNet):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnety120_regnety_120.sw_in12k_ft_in1k.keras",
        )
    ]

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
        name: str = "RegNetY120",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            168,
            73.36,
            2.37,
            112,
            19,
            0.25,
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


class RegNetX160(RegNet):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnetx160_regnetx_160.tv2_in1k.keras",
        )
    ]

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
        name: str = "RegNetX160",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            216,
            55.59,
            2.1,
            128,
            22,
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


class RegNetY160(RegNet):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnety160_regnety_160.swag_ft_in1k.keras",
        )
    ]

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
        name: str = "RegNetY160",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            200,
            106.23,
            2.48,
            112,
            18,
            0.25,
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


class RegNetX320(RegNet):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnetx320_regnetx_320.tv2_in1k.keras",
        )
    ]

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
        name: str = "RegNetX320",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            320,
            69.86,
            2.0,
            168,
            23,
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


class RegNetY320(RegNet):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnety320_regnety_320.swag_ft_in1k.keras",
        )
    ]

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
        name: str = "RegNetY320",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            232,
            115.89,
            2.53,
            232,
            20,
            0.25,
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


add_model_to_registry(RegNetX002, "imagenet")
add_model_to_registry(RegNetY002, "imagenet")
add_model_to_registry(RegNetX004, "imagenet")
add_model_to_registry(RegNetY004, "imagenet")
add_model_to_registry(RegNetX006, "imagenet")
add_model_to_registry(RegNetY006, "imagenet")
add_model_to_registry(RegNetX008, "imagenet")
add_model_to_registry(RegNetY008, "imagenet")
add_model_to_registry(RegNetX016, "imagenet")
add_model_to_registry(RegNetY016, "imagenet")
add_model_to_registry(RegNetX032, "imagenet")
add_model_to_registry(RegNetY032, "imagenet")
add_model_to_registry(RegNetX040, "imagenet")
add_model_to_registry(RegNetY040, "imagenet")
add_model_to_registry(RegNetX064, "imagenet")
add_model_to_registry(RegNetY064, "imagenet")
add_model_to_registry(RegNetX080, "imagenet")
add_model_to_registry(RegNetY080, "imagenet")
add_model_to_registry(RegNetX120, "imagenet")
add_model_to_registry(RegNetY120, "imagenet")
add_model_to_registry(RegNetX160, "imagenet")
add_model_to_registry(RegNetY160, "imagenet")
add_model_to_registry(RegNetX320, "imagenet")
add_model_to_registry(RegNetY320, "imagenet")

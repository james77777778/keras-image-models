import typing

import keras
import numpy as np
from keras import backend
from keras import layers

from kimm._src.blocks.conv2d import apply_conv2d_block
from kimm._src.blocks.squeeze_and_excitation import apply_se_block
from kimm._src.kimm_export import kimm_export
from kimm._src.models.base_model import BaseModel
from kimm._src.utils.model_registry import add_model_to_registry


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


# Model Definition


class RegNetVariant(RegNet):
    # Parameters
    w0 = None
    wa = None
    wm = None
    group_size = None
    depth = None
    se_ratio = None

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
        if type(self) is RegNetVariant:
            raise NotImplementedError(
                f"Cannot instantiate base class: {self.__class__.__name__}. "
                "You should use its subclasses."
            )
        kwargs = self.fix_config(kwargs)
        super().__init__(
            w0=self.w0,
            wa=self.wa,
            wm=self.wm,
            group_size=self.group_size,
            depth=self.depth,
            se_ratio=self.se_ratio,
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


@kimm_export(parent_path=["kimm.models", "kimm.models.regnet"])
class RegNetX002(RegNetVariant):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnetx002_regnetx_002.pycls_in1k.keras",
        )
    ]

    # Parameters
    w0 = 24
    wa = 36.44
    wm = 2.49
    group_size = 8
    depth = 13
    se_ratio = 0.0


@kimm_export(parent_path=["kimm.models", "kimm.models.regnet"])
class RegNetY002(RegNetVariant):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnety002_regnety_002.pycls_in1k.keras",
        )
    ]

    # Parameters
    w0 = 24
    wa = 36.44
    wm = 2.49
    group_size = 8
    depth = 13
    se_ratio = 0.25


@kimm_export(parent_path=["kimm.models", "kimm.models.regnet"])
class RegNetX004(RegNetVariant):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnetx004_regnetx_004.pycls_in1k.keras",
        )
    ]

    # Parameters
    w0 = 24
    wa = 24.48
    wm = 2.54
    group_size = 16
    depth = 22
    se_ratio = 0.0


@kimm_export(parent_path=["kimm.models", "kimm.models.regnet"])
class RegNetY004(RegNetVariant):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnety004_regnety_004.tv2_in1k.keras",
        )
    ]

    # Parameters
    w0 = 48
    wa = 27.89
    wm = 2.09
    group_size = 8
    depth = 16
    se_ratio = 0.25


@kimm_export(parent_path=["kimm.models", "kimm.models.regnet"])
class RegNetX006(RegNetVariant):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnetx006_regnetx_006.pycls_in1k.keras",
        )
    ]

    # Parameters
    w0 = 48
    wa = 36.97
    wm = 2.24
    group_size = 24
    depth = 16
    se_ratio = 0.0


@kimm_export(parent_path=["kimm.models", "kimm.models.regnet"])
class RegNetY006(RegNetVariant):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnety006_regnety_006.pycls_in1k.keras",
        )
    ]

    # Parameters
    w0 = 48
    wa = 32.54
    wm = 2.32
    group_size = 16
    depth = 15
    se_ratio = 0.25


@kimm_export(parent_path=["kimm.models", "kimm.models.regnet"])
class RegNetX008(RegNetVariant):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnetx008_regnetx_008.tv2_in1k.keras",
        )
    ]

    # Parameters
    w0 = 56
    wa = 35.73
    wm = 2.28
    group_size = 16
    depth = 16
    se_ratio = 0.0


@kimm_export(parent_path=["kimm.models", "kimm.models.regnet"])
class RegNetY008(RegNetVariant):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnety008_regnety_008.pycls_in1k.keras",
        )
    ]

    # Parameters
    w0 = 56
    wa = 38.84
    wm = 2.4
    group_size = 16
    depth = 14
    se_ratio = 0.25


@kimm_export(parent_path=["kimm.models", "kimm.models.regnet"])
class RegNetX016(RegNetVariant):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnetx016_regnetx_016.tv2_in1k.keras",
        )
    ]

    # Parameters
    w0 = 80
    wa = 34.01
    wm = 2.25
    group_size = 24
    depth = 18
    se_ratio = 0.0


@kimm_export(parent_path=["kimm.models", "kimm.models.regnet"])
class RegNetY016(RegNetVariant):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnety016_regnety_016.tv2_in1k.keras",
        )
    ]

    # Parameters
    w0 = 48
    wa = 20.71
    wm = 2.65
    group_size = 24
    depth = 27
    se_ratio = 0.25


@kimm_export(parent_path=["kimm.models", "kimm.models.regnet"])
class RegNetX032(RegNetVariant):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnetx032_regnetx_032.tv2_in1k.keras",
        )
    ]

    # Parameters
    w0 = 88
    wa = 26.31
    wm = 2.25
    group_size = 48
    depth = 25
    se_ratio = 0.0


@kimm_export(parent_path=["kimm.models", "kimm.models.regnet"])
class RegNetY032(RegNetVariant):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnety032_regnety_032.ra_in1k.keras",
        )
    ]

    # Parameters
    w0 = 80
    wa = 42.63
    wm = 2.66
    group_size = 24
    depth = 21
    se_ratio = 0.25


@kimm_export(parent_path=["kimm.models", "kimm.models.regnet"])
class RegNetX040(RegNetVariant):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnetx040_regnetx_040.pycls_in1k.keras",
        )
    ]

    # Parameters
    w0 = 96
    wa = 38.65
    wm = 2.43
    group_size = 40
    depth = 23
    se_ratio = 0.0


@kimm_export(parent_path=["kimm.models", "kimm.models.regnet"])
class RegNetY040(RegNetVariant):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnety040_regnety_040.ra3_in1k.keras",
        )
    ]

    # Parameters
    w0 = 96
    wa = 31.41
    wm = 2.24
    group_size = 64
    depth = 22
    se_ratio = 0.25


@kimm_export(parent_path=["kimm.models", "kimm.models.regnet"])
class RegNetX064(RegNetVariant):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnetx064_regnetx_064.pycls_in1k.keras",
        )
    ]

    # Parameters
    w0 = 184
    wa = 60.83
    wm = 2.07
    group_size = 56
    depth = 17
    se_ratio = 0.0


@kimm_export(parent_path=["kimm.models", "kimm.models.regnet"])
class RegNetY064(RegNetVariant):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnety064_regnety_064.ra3_in1k.keras",
        )
    ]

    # Parameters
    w0 = 112
    wa = 33.22
    wm = 2.27
    group_size = 72
    depth = 25
    se_ratio = 0.25


@kimm_export(parent_path=["kimm.models", "kimm.models.regnet"])
class RegNetX080(RegNetVariant):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnetx080_regnetx_080.tv2_in1k.keras",
        )
    ]

    # Parameters
    w0 = 80
    wa = 49.56
    wm = 2.88
    group_size = 120
    depth = 23
    se_ratio = 0.0


@kimm_export(parent_path=["kimm.models", "kimm.models.regnet"])
class RegNetY080(RegNetVariant):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnety080_regnety_080.ra3_in1k.keras",
        )
    ]

    # Parameters
    w0 = 192
    wa = 76.82
    wm = 2.19
    group_size = 56
    depth = 17
    se_ratio = 0.25


@kimm_export(parent_path=["kimm.models", "kimm.models.regnet"])
class RegNetX120(RegNetVariant):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnetx120_regnetx_120.pycls_in1k.keras",
        )
    ]

    # Parameters
    w0 = 168
    wa = 73.36
    wm = 2.37
    group_size = 112
    depth = 19
    se_ratio = 0.0


@kimm_export(parent_path=["kimm.models", "kimm.models.regnet"])
class RegNetY120(RegNetVariant):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnety120_regnety_120.sw_in12k_ft_in1k.keras",
        )
    ]

    # Parameters
    w0 = 168
    wa = 73.36
    wm = 2.37
    group_size = 112
    depth = 19
    se_ratio = 0.25


@kimm_export(parent_path=["kimm.models", "kimm.models.regnet"])
class RegNetX160(RegNetVariant):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnetx160_regnetx_160.tv2_in1k.keras",
        )
    ]

    # Parameters
    w0 = 216
    wa = 55.59
    wm = 2.1
    group_size = 128
    depth = 22
    se_ratio = 0.0


@kimm_export(parent_path=["kimm.models", "kimm.models.regnet"])
class RegNetY160(RegNetVariant):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnety160_regnety_160.swag_ft_in1k.keras",
        )
    ]

    # Parameters
    w0 = 200
    wa = 106.23
    wm = 2.48
    group_size = 112
    depth = 18
    se_ratio = 0.25


@kimm_export(parent_path=["kimm.models", "kimm.models.regnet"])
class RegNetX320(RegNetVariant):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnetx320_regnetx_320.tv2_in1k.keras",
        )
    ]

    # Parameters
    w0 = 320
    wa = 69.86
    wm = 2.0
    group_size = 168
    depth = 23
    se_ratio = 0.0


@kimm_export(parent_path=["kimm.models", "kimm.models.regnet"])
class RegNetY320(RegNetVariant):
    available_weights = [
        (
            "imagenet",
            RegNet.default_origin,
            "regnety320_regnety_320.swag_ft_in1k.keras",
        )
    ]

    # Parameters
    w0 = 232
    wa = 115.89
    wm = 2.53
    group_size = 232
    depth = 20
    se_ratio = 0.25


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

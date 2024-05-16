import typing

import keras
from keras import backend
from keras import layers

from kimm._src.blocks.conv2d import apply_conv2d_block
from kimm._src.kimm_export import kimm_export
from kimm._src.layers.learnable_affine import LearnableAffine
from kimm._src.models.base_model import BaseModel
from kimm._src.utils.model_registry import add_model_to_registry

DEFAULT_V1_TINY_CONFIG = dict(
    stem_type="v1",
    stem_channels=[48, 48, 96],
    use_learnable_affine=False,
    # input_channels, hidden_channels, output_channels, blocks, downsample,
    # light_block, kernel_size, num_layers
    stage1=[96, 96, 224, 1, False, False, 3, 5],
    stage2=[224, 128, 448, 1, True, False, 3, 5],
    stage3=[448, 160, 512, 2, True, False, 3, 5],
    stage4=[512, 192, 768, 1, True, False, 3, 5],
)
DEFAULT_V1_SMALL_CONFIG = dict(
    stem_type="v1",
    stem_channels=[64, 64, 128],
    use_learnable_affine=False,
    # input_channels, hidden_channels, output_channels, blocks, downsample,
    # light_block, kernel_size, num_layers
    stage1=[128, 128, 256, 1, False, False, 3, 6],
    stage2=[256, 160, 512, 1, True, False, 3, 6],
    stage3=[512, 192, 768, 2, True, False, 3, 6],
    stage4=[768, 224, 1024, 1, True, False, 3, 6],
)
DEFAULT_V1_BASE_CONFIG = dict(
    stem_type="v1",
    stem_channels=[96, 96, 160],
    use_learnable_affine=False,
    # input_channels, hidden_channels, output_channels, blocks, downsample,
    # light_block, kernel_size, num_layers
    stage1=[160, 192, 320, 1, False, False, 3, 7],
    stage2=[320, 224, 640, 2, True, False, 3, 7],
    stage3=[640, 256, 960, 3, True, False, 3, 7],
    stage4=[960, 288, 1280, 2, True, False, 3, 7],
)
DEFAULT_V2_B0_CONFIG = dict(
    stem_type="v2",
    stem_channels=[16, 16],
    use_learnable_affine=True,
    # input_channels, hidden_channels, output_channels, blocks, downsample,
    # light_block, kernel_size, num_layers
    stage1=[16, 16, 64, 1, False, False, 3, 3],
    stage2=[64, 32, 256, 1, True, False, 3, 3],
    stage3=[256, 64, 512, 2, True, True, 5, 3],
    stage4=[512, 128, 1024, 1, True, True, 5, 3],
)
DEFAULT_V2_B1_CONFIG = dict(
    stem_type="v2",
    stem_channels=[24, 32],
    use_learnable_affine=True,
    # input_channels, hidden_channels, output_channels, blocks, downsample,
    # light_block, kernel_size, num_layers
    stage1=[32, 32, 64, 1, False, False, 3, 3],
    stage2=[64, 48, 256, 1, True, False, 3, 3],
    stage3=[256, 96, 512, 2, True, True, 5, 3],
    stage4=[512, 192, 1024, 1, True, True, 5, 3],
)
DEFAULT_V2_B2_CONFIG = dict(
    stem_type="v2",
    stem_channels=[24, 32],
    use_learnable_affine=True,
    # input_channels, hidden_channels, output_channels, blocks, downsample,
    # light_block, kernel_size, num_layers
    stage1=[32, 32, 96, 1, False, False, 3, 4],
    stage2=[96, 64, 384, 1, True, False, 3, 4],
    stage3=[384, 128, 768, 3, True, True, 5, 4],
    stage4=[768, 256, 1536, 1, True, True, 5, 4],
)
DEFAULT_V2_B3_CONFIG = dict(
    stem_type="v2",
    stem_channels=[24, 32],
    use_learnable_affine=True,
    # input_channels, hidden_channels, output_channels, blocks, downsample,
    # light_block, kernel_size, num_layers
    stage1=[32, 32, 128, 1, False, False, 3, 5],
    stage2=[128, 64, 512, 1, True, False, 3, 5],
    stage3=[512, 128, 1024, 3, True, True, 5, 5],
    stage4=[1024, 256, 2048, 1, True, True, 5, 5],
)
DEFAULT_V2_B4_CONFIG = dict(
    stem_type="v2",
    stem_channels=[32, 48],
    use_learnable_affine=False,
    # input_channels, hidden_channels, output_channels, blocks, downsample,
    # light_block, kernel_size, num_layers
    stage1=[48, 48, 128, 1, False, False, 3, 6],
    stage2=[128, 96, 512, 1, True, False, 3, 6],
    stage3=[512, 192, 1024, 3, True, True, 5, 6],
    stage4=[1024, 384, 2048, 1, True, True, 5, 6],
)
DEFAULT_V2_B5_CONFIG = dict(
    stem_type="v2",
    stem_channels=[32, 64],
    use_learnable_affine=False,
    # input_channels, hidden_channels, output_channels, blocks, downsample,
    # light_block, kernel_size, num_layers
    stage1=[64, 64, 128, 1, False, False, 3, 6],
    stage2=[128, 128, 512, 2, True, False, 3, 6],
    stage3=[512, 256, 1024, 5, True, True, 5, 6],
    stage4=[1024, 512, 2048, 2, True, True, 5, 6],
)
DEFAULT_V2_B6_CONFIG = dict(
    stem_type="v2",
    stem_channels=[48, 96],
    use_learnable_affine=False,
    # input_channels, hidden_channels, output_channels, blocks, downsample,
    # light_block, kernel_size, num_layers
    stage1=[96, 96, 192, 2, False, False, 3, 6],
    stage2=[192, 192, 512, 3, True, False, 3, 6],
    stage3=[512, 384, 1024, 6, True, True, 5, 6],
    stage4=[1024, 768, 2048, 3, True, True, 5, 6],
)


def apply_conv_bn_act_block(
    inputs,
    filters,
    kernel_size,
    strides=1,
    activation="relu",
    use_depthwise=False,
    padding=None,
    use_learnable_affine=False,
    name="conv_bn_act_block",
):
    x = inputs
    x = apply_conv2d_block(
        x,
        filters,
        kernel_size,
        strides,
        activation=activation,
        use_depthwise=use_depthwise,
        padding=padding,
        name=name,
    )
    if activation is not None and use_learnable_affine:
        x = LearnableAffine(name=f"{name}_lab")(x)
    return x


def apply_stem_v1(inputs, stem_channels, name="stem_v1"):
    x = inputs
    for i, c in enumerate(stem_channels):
        x = apply_conv_bn_act_block(
            x, c, 3, strides=2 if i == 0 else 1, name=f"{name}_{i}"
        )
    x = layers.ZeroPadding2D(padding=1)(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)
    return x


def apply_stem_v2(
    inputs,
    hidden_channels,
    output_channels,
    use_learnable_affine=False,
    name="stem_v2",
):
    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3

    x = inputs
    x = apply_conv_bn_act_block(
        x,
        hidden_channels,
        3,
        2,
        use_learnable_affine=use_learnable_affine,
        name=f"{name}_stem1",
    )
    x = layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(x)

    x2 = apply_conv_bn_act_block(
        x,
        hidden_channels // 2,
        2,
        1,
        padding="valid",
        use_learnable_affine=use_learnable_affine,
        name=f"{name}_stem2a",
    )
    x2 = layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(x2)
    x2 = apply_conv_bn_act_block(
        x2,
        hidden_channels,
        2,
        1,
        padding="valid",
        use_learnable_affine=use_learnable_affine,
        name=f"{name}_stem2b",
    )

    x1 = layers.MaxPooling2D(pool_size=2, strides=1)(x)
    x = layers.Concatenate(axis=channels_axis)([x1, x2])
    x = apply_conv_bn_act_block(
        x,
        hidden_channels,
        3,
        2,
        use_learnable_affine=use_learnable_affine,
        name=f"{name}_stem3",
    )
    x = apply_conv_bn_act_block(
        x,
        output_channels,
        1,
        1,
        use_learnable_affine=use_learnable_affine,
        name=f"{name}_stem4",
    )
    return x


def apply_light_conv_bn_act_block(
    inputs,
    filters,
    kernel_size,
    use_learnable_affine=False,
    name="light_conv_bn_act_block",
):
    x = inputs
    x = apply_conv_bn_act_block(
        x,
        filters,
        1,
        activation=None,
        use_learnable_affine=use_learnable_affine,
        name=f"{name}_conv1",
    )
    x = apply_conv_bn_act_block(
        x,
        filters,
        kernel_size,
        activation="relu",
        use_depthwise=True,
        use_learnable_affine=use_learnable_affine,
        name=f"{name}_conv2",
    )
    return x


def apply_ese_module(inputs, channels, name="ese_module"):
    x = inputs
    x = layers.GlobalAveragePooling2D(keepdims=True)(x)
    x = layers.Conv2D(
        channels, 1, 1, "valid", use_bias=True, name=f"{name}_conv"
    )(x)
    x = layers.Activation("sigmoid")(x)
    x = layers.Multiply()([inputs, x])
    return x


def apply_high_perf_gpu_block(
    inputs,
    num_layers,
    hidden_channels,
    output_channels,
    kernel_size,
    add_skip=False,
    use_light_block=False,
    use_learnable_affine=False,
    aggregation="ese",
    name="high_perf_gpu_block",
):
    if aggregation not in ("se", "ese"):
        raise ValueError(
            "aggregation must be one of ('se', 'ese'). "
            f"Receviced: aggregation={aggregation}"
        )
    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3

    x = inputs
    outputs = [x]
    for i in range(num_layers):
        if use_light_block:
            x = apply_light_conv_bn_act_block(
                x,
                hidden_channels,
                kernel_size,
                use_learnable_affine=use_learnable_affine,
                name=f"{name}_layers_{i}",
            )
        else:
            x = apply_conv_bn_act_block(
                x,
                hidden_channels,
                kernel_size,
                strides=1,
                use_learnable_affine=use_learnable_affine,
                name=f"{name}_layers_{i}",
            )
        outputs.append(x)
    x = layers.Concatenate(axis=channels_axis)(outputs)
    if aggregation == "se":
        x = apply_conv_bn_act_block(
            x,
            output_channels // 2,
            1,
            1,
            use_learnable_affine=use_learnable_affine,
            name=f"{name}_aggregation_0",
        )
        x = apply_conv_bn_act_block(
            x,
            output_channels,
            1,
            1,
            use_learnable_affine=use_learnable_affine,
            name=f"{name}_aggregation_1",
        )
    else:
        x = apply_conv_bn_act_block(
            x,
            output_channels,
            1,
            1,
            use_learnable_affine=use_learnable_affine,
            name=f"{name}_aggregation_0",
        )
        x = apply_ese_module(x, output_channels, name=f"{name}_aggregation_1")
    if add_skip:
        x = layers.Add()([x, inputs])
    return x


def apply_high_perf_gpu_stage(
    inputs,
    num_blocks,
    num_layers,
    hidden_channels,
    output_channels,
    kernel_size=3,
    strides=2,
    downsample=True,
    use_light_block=False,
    use_learnable_affine=False,
    aggregation="ese",
    name="high_perf_gpu_stage",
):
    if aggregation not in ("se", "ese"):
        raise ValueError(
            "aggregation must be one of ('se', 'ese'). "
            f"Receviced: aggregation={aggregation}"
        )
    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3
    input_channels = inputs.shape[channels_axis]

    x = inputs
    if downsample:
        x = apply_conv_bn_act_block(
            x,
            input_channels,
            3,
            strides,
            activation=None,
            use_depthwise=True,
            use_learnable_affine=use_learnable_affine,
            name=f"{name}_downsample",
        )
    for i in range(num_blocks):
        x = apply_high_perf_gpu_block(
            x,
            num_layers,
            hidden_channels,
            output_channels,
            kernel_size,
            add_skip=False if i == 0 else True,
            use_light_block=use_light_block,
            use_learnable_affine=use_learnable_affine,
            aggregation=aggregation,
            name=f"{name}_blocks_{i}",
        )
    return x


@keras.saving.register_keras_serializable(package="kimm")
class HGNet(BaseModel):
    available_feature_keys = [
        "STEM_S4",
        *[f"BLOCK{i}_S{j}" for i, j in zip(range(4), [4, 8, 16, 32])],
    ]

    def __init__(
        self,
        config: str = "v1_tiny",
        **kwargs,
    ):
        kwargs["weights_url"] = self.get_weights_url(kwargs["weights"])

        _available_configs = ["v1_tiny", "v1_small", "v1_base"]
        if config == "v1_tiny":
            _config = DEFAULT_V1_TINY_CONFIG
        elif config == "v1_small":
            _config = DEFAULT_V1_SMALL_CONFIG
        elif config == "v1_base":
            _config = DEFAULT_V1_BASE_CONFIG
        elif config == "v2_b0":
            _config = DEFAULT_V2_B0_CONFIG
        elif config == "v2_b1":
            _config = DEFAULT_V2_B1_CONFIG
        elif config == "v2_b2":
            _config = DEFAULT_V2_B2_CONFIG
        elif config == "v2_b3":
            _config = DEFAULT_V2_B3_CONFIG
        elif config == "v2_b4":
            _config = DEFAULT_V2_B4_CONFIG
        elif config == "v2_b5":
            _config = DEFAULT_V2_B5_CONFIG
        elif config == "v2_b6":
            _config = DEFAULT_V2_B6_CONFIG
        else:
            raise ValueError(
                f"config must be one of {_available_configs} using string. "
                f"Received: config={config}"
            )

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
        use_learnable_affine = _config["use_learnable_affine"]
        stem_channels = _config["stem_channels"]
        if _config["stem_type"] == "v1":
            x = apply_stem_v1(x, stem_channels, name="stem")
        elif _config["stem_type"] == "v2":
            x = apply_stem_v2(
                x,
                stem_channels[0],
                stem_channels[1],
                use_learnable_affine=use_learnable_affine,
                name="stem",
            )
        else:
            raise NotImplementedError
        features["STEM_S4"] = x

        # stages
        current_stride = 4
        stage_config = [
            _config["stage1"],
            _config["stage2"],
            _config["stage3"],
            _config["stage4"],
        ]
        for current_stage_idx, (_, h, o, b, d, light, k, n) in enumerate(
            stage_config
        ):
            x = apply_high_perf_gpu_stage(
                x,
                num_blocks=b,
                num_layers=n,
                hidden_channels=h,
                output_channels=o,
                kernel_size=k,
                strides=2,
                downsample=d,
                use_light_block=light,
                use_learnable_affine=use_learnable_affine,
                aggregation="ese" if _config["stem_type"] == "v1" else "se",
                name=f"stages_{current_stage_idx}",
            )
            if d:
                current_stride *= 2
            # add feature
            features[f"BLOCK{current_stage_idx}_S{current_stride}"] = x

        # Head
        x = self.build_head(x, use_learnable_affine=use_learnable_affine)

        super().__init__(inputs=inputs, outputs=x, features=features, **kwargs)

        # All references to `self` below this line
        self.config = config

    def build_top(
        self,
        inputs,
        classes: int,
        classifier_activation: str,
        dropout_rate: float,
        use_learnable_affine: bool = False,
    ):
        class_expand = 2048
        x = layers.GlobalAveragePooling2D(name="avg_pool", keepdims=True)(
            inputs
        )
        x = layers.Conv2D(
            class_expand,
            1,
            1,
            "valid",
            activation="relu",
            use_bias=False,
            name="head_last_conv_0",
        )(x)
        if use_learnable_affine:
            x = LearnableAffine(name="head_last_conv_2")(x)
        x = layers.Dropout(rate=dropout_rate, name="head_dropout")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(
            classes, activation=classifier_activation, name="classifier"
        )(x)
        return x

    def build_head(self, inputs, use_learnable_affine=False):
        x = inputs
        if self._include_top:
            x = self.build_top(
                x,
                self._classes,
                self._classifier_activation,
                self._dropout_rate,
                use_learnable_affine=use_learnable_affine,
            )
        else:
            if self._pooling == "avg":
                x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            elif self._pooling == "max":
                x = layers.GlobalMaxPooling2D(name="max_pool")(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"config": self.config})
        return config

    def fix_config(self, config):
        unused_kwargs = ["config"]
        for k in unused_kwargs:
            config.pop(k, None)
        return config


# Model Definition


class HGNetVariant(HGNet):
    # Parameters
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
        if type(self) is HGNetVariant:
            raise NotImplementedError(
                f"Cannot instantiate base class: {self.__class__.__name__}. "
                "You should use its subclasses."
            )
        kwargs = self.fix_config(kwargs)
        super().__init__(
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


@kimm_export(parent_path=["kimm.models", "kimm.models.hgnet"])
class HGNetTiny(HGNetVariant):
    available_weights = [
        (
            "imagenet",
            HGNet.default_origin,
            "hgnettiny_hgnet_tiny.ssld_in1k.keras",
        )
    ]

    # Parameters
    config = "v1_tiny"


@kimm_export(parent_path=["kimm.models", "kimm.models.hgnet"])
class HGNetSmall(HGNetVariant):
    available_weights = [
        (
            "imagenet",
            HGNet.default_origin,
            "hgnetsmall_hgnet_small.ssld_in1k.keras",
        )
    ]

    # Parameters
    config = "v1_small"


@kimm_export(parent_path=["kimm.models", "kimm.models.hgnet"])
class HGNetBase(HGNetVariant):
    available_weights = [
        (
            "imagenet",
            HGNet.default_origin,
            "hgnetbase_hgnet_base.ssld_in1k.keras",
        )
    ]

    # Parameters
    config = "v1_base"


@kimm_export(parent_path=["kimm.models", "kimm.models.hgnet"])
class HGNetV2B0(HGNetVariant):
    available_weights = [
        (
            "imagenet",
            HGNet.default_origin,
            "hgnetv2b0_hgnetv2_b0.ssld_stage2_ft_in1k.keras",
        )
    ]

    # Parameters
    config = "v2_b0"


@kimm_export(parent_path=["kimm.models", "kimm.models.hgnet"])
class HGNetV2B1(HGNetVariant):
    available_weights = [
        (
            "imagenet",
            HGNet.default_origin,
            "hgnetv2b1_hgnetv2_b1.ssld_stage2_ft_in1k.keras",
        )
    ]

    # Parameters
    config = "v2_b1"


@kimm_export(parent_path=["kimm.models", "kimm.models.hgnet"])
class HGNetV2B2(HGNetVariant):
    available_weights = [
        (
            "imagenet",
            HGNet.default_origin,
            "hgnetv2b2_hgnetv2_b2.ssld_stage2_ft_in1k.keras",
        )
    ]

    # Parameters
    config = "v2_b2"


@kimm_export(parent_path=["kimm.models", "kimm.models.hgnet"])
class HGNetV2B3(HGNetVariant):
    available_weights = [
        (
            "imagenet",
            HGNet.default_origin,
            "hgnetv2b3_hgnetv2_b3.ssld_stage2_ft_in1k.keras",
        )
    ]

    # Parameters
    config = "v2_b3"


@kimm_export(parent_path=["kimm.models", "kimm.models.hgnet"])
class HGNetV2B4(HGNetVariant):
    available_weights = [
        (
            "imagenet",
            HGNet.default_origin,
            "hgnetv2b4_hgnetv2_b4.ssld_stage2_ft_in1k.keras",
        )
    ]

    # Parameters
    config = "v2_b4"


@kimm_export(parent_path=["kimm.models", "kimm.models.hgnet"])
class HGNetV2B5(HGNetVariant):
    available_weights = [
        (
            "imagenet",
            HGNet.default_origin,
            "hgnetv2b5_hgnetv2_b5.ssld_stage2_ft_in1k.keras",
        )
    ]

    # Parameters
    config = "v2_b5"


@kimm_export(parent_path=["kimm.models", "kimm.models.hgnet"])
class HGNetV2B6(HGNetVariant):
    available_weights = [
        (
            "imagenet",
            HGNet.default_origin,
            "hgnetv2b6_hgnetv2_b6.ssld_stage2_ft_in1k.keras",
        )
    ]

    # Parameters
    config = "v2_b6"


add_model_to_registry(HGNetTiny, "imagenet")
add_model_to_registry(HGNetSmall, "imagenet")
add_model_to_registry(HGNetBase, "imagenet")
add_model_to_registry(HGNetV2B0, "imagenet")
add_model_to_registry(HGNetV2B1, "imagenet")
add_model_to_registry(HGNetV2B2, "imagenet")
add_model_to_registry(HGNetV2B3, "imagenet")
add_model_to_registry(HGNetV2B4, "imagenet")
add_model_to_registry(HGNetV2B5, "imagenet")
add_model_to_registry(HGNetV2B6, "imagenet")

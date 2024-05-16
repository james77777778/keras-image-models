import typing

import keras
from keras import backend
from keras import initializers
from keras import layers

from kimm._src.blocks.transformer import apply_mlp_block
from kimm._src.kimm_export import kimm_export
from kimm._src.layers.layer_scale import LayerScale
from kimm._src.models.base_model import BaseModel
from kimm._src.utils.model_registry import add_model_to_registry


def apply_convnext_block(
    inputs,
    output_channels,
    kernel_size,
    strides,
    mlp_ratio,
    activation="gelu",
    use_conv_mlp=False,
    use_grn=False,
    name="convnext_block",
):
    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3
    input_channels = inputs.shape[channels_axis]
    hidden_channels = int(mlp_ratio * output_channels)

    x = inputs
    shortcut = inputs

    # Padding
    padding = "same"
    if strides > 1:
        padding = "valid"
        x = layers.ZeroPadding2D(
            (kernel_size[0] // 2, kernel_size[1] // 2), name=f"{name}_pad"
        )(x)

    # Depthwise
    x = layers.DepthwiseConv2D(
        kernel_size,
        strides,
        padding=padding,
        use_bias=True,
        name=f"{name}_conv_dw_dwconv2d",
    )(x)
    x = layers.LayerNormalization(
        axis=channels_axis, epsilon=1e-6, name=f"{name}_norm"
    )(x)

    # MLP
    x = apply_mlp_block(
        x,
        hidden_channels,
        output_channels,
        activation,
        use_bias=True,
        use_conv_mlp=use_conv_mlp,
        name=f"{name}_mlp",
    )

    # LayerScale
    x = LayerScale(
        axis=channels_axis,
        initializer=initializers.Constant(1e-6),
        name=f"{name}_layerscale",
    )(x)

    # Downsample
    if input_channels != output_channels or strides != 1:
        shortcut = layers.AveragePooling2D(
            2, strides=strides, name=f"{name}_pool"
        )(shortcut)
        if input_channels != output_channels:
            shortcut = layers.Conv2D(
                output_channels, 1, 1, use_bias=False, name=f"{name}_conv"
            )(shortcut)

    x = layers.Add()([x, shortcut])
    return x


def apply_convnext_stage(
    inputs,
    depth: int,
    output_channels: int,
    kernel_size: int,
    strides: int,
    activation="gelu",
    use_conv_mlp=False,
    use_grn=False,
    name="convnext_stage",
):
    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3
    input_channels = inputs.shape[channels_axis]

    x = inputs

    # Downsample
    if input_channels != output_channels or strides > 1:
        ds_ks = 2 if strides > 1 else 1
        x = layers.LayerNormalization(
            axis=channels_axis, epsilon=1e-6, name=f"{name}_downsample_0"
        )(x)
        x = layers.Conv2D(
            output_channels,
            ds_ks,
            strides,
            padding="valid",
            use_bias=True,
            name=f"{name}_downsample_1_conv2d",
        )(x)

    for i in range(depth):
        x = apply_convnext_block(
            x,
            output_channels,
            kernel_size,
            1,
            mlp_ratio=4,
            activation=activation,
            use_conv_mlp=use_conv_mlp,
            use_grn=use_grn,
            name=f"{name}_blocks_{i}",
        )
    return x


@keras.saving.register_keras_serializable(package="kimm")
class ConvNeXt(BaseModel):
    available_feature_keys = [
        "STEM_S4",
        *[f"BLOCK{i}_S{2**(i+2)}" for i in range(4)],
    ]

    def __init__(
        self,
        depths: typing.Sequence[int] = [3, 3, 9, 3],
        hidden_channels: typing.Sequence[int] = [96, 192, 384, 768],
        patch_size: int = 4,
        kernel_size: int = 7,
        activation: str = "gelu",
        use_conv_mlp: bool = False,
        **kwargs,
    ):
        kwargs["weights_url"] = self.get_weights_url(kwargs["weights"])
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

        # Stem
        x = layers.Conv2D(
            hidden_channels[0],
            patch_size,
            patch_size,
            use_bias=True,
            name="stem_0_conv2d",
        )(x)
        x = layers.LayerNormalization(
            axis=channels_axis, epsilon=1e-6, name="stem_1"
        )(x)
        features["STEM_S4"] = x

        # Blocks (4 stages)
        current_stride = patch_size
        for i in range(4):
            strides = 2 if i > 0 else 1
            x = apply_convnext_stage(
                x,
                depths[i],
                hidden_channels[i],
                kernel_size,
                strides,
                activation,
                use_conv_mlp,
                use_grn=False,
                name=f"stages_{i}",
            )
            current_stride *= strides
            # Add feature
            features[f"BLOCK{i}_S{current_stride}"] = x

        # Head
        x = self.build_head(x)

        super().__init__(inputs=inputs, outputs=x, features=features, **kwargs)

        # All references to `self` below this line
        self.depths = depths
        self.hidden_channels = hidden_channels
        self.patch_size = patch_size
        self.kernel_size = kernel_size
        self.activation = activation
        self.use_conv_mlp = use_conv_mlp

    def build_top(self, inputs, classes, classifier_activation, dropout_rate):
        x = inputs
        x = layers.GlobalAveragePooling2D(name="avg_pool")(inputs)
        x = layers.LayerNormalization(axis=-1, epsilon=1e-6, name="head_norm")(
            x
        )
        x = layers.Dropout(rate=dropout_rate, name="head_dropout")(x)
        x = layers.Dense(
            classes, activation=classifier_activation, name="classifier"
        )(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "depths": self.depths,
                "hidden_channels": self.hidden_channels,
                "patch_size": self.patch_size,
                "kernel_size": self.kernel_size,
                "activation": self.activation,
                "use_conv_mlp": self.use_conv_mlp,
            }
        )
        return config

    def fix_config(self, config):
        unused_kwargs = [
            "depths",
            "hidden_channels",
            "patch_size",
            "kernel_size",
            "activation",
            "use_conv_mlp",
        ]
        for k in unused_kwargs:
            config.pop(k, None)
        return config


# Model Definition


class ConvNeXtVariant(ConvNeXt):
    # Parameters
    depths = None
    hidden_channels = None
    patch_size = None
    kernel_size = None
    activation = None
    use_conv_mlp = None

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
        if type(self) is ConvNeXtVariant:
            raise NotImplementedError(
                f"Cannot instantiate base class: {self.__class__.__name__}. "
                "You should use its subclasses."
            )
        kwargs = self.fix_config(kwargs)
        if len(getattr(self, "available_weights", [])) == 0:
            weights = None
        super().__init__(
            depths=self.depths,
            hidden_channels=self.hidden_channels,
            patch_size=self.patch_size,
            kernel_size=self.kernel_size,
            activation=self.activation,
            use_conv_mlp=self.use_conv_mlp,
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


@kimm_export(parent_path=["kimm.models", "kimm.models.convnext"])
class ConvNeXtAtto(ConvNeXtVariant):
    available_weights = [
        (
            "imagenet",
            ConvNeXt.default_origin,
            "convnextatto_convnext_atto.d2_in1k.keras",
        )
    ]

    # Parameters
    depths = (2, 2, 6, 2)
    hidden_channels = (40, 80, 160, 320)
    patch_size = 4
    kernel_size = 7
    activation = "gelu"
    use_conv_mlp = True


@kimm_export(parent_path=["kimm.models", "kimm.models.convnext"])
class ConvNeXtFemto(ConvNeXtVariant):
    available_weights = [
        (
            "imagenet",
            ConvNeXt.default_origin,
            "convnextfemto_convnext_femto.d1_in1k.keras",
        )
    ]

    # Parameters
    depths = (2, 2, 6, 2)
    hidden_channels = (48, 96, 192, 384)
    patch_size = 4
    kernel_size = 7
    activation = "gelu"
    use_conv_mlp = True


@kimm_export(parent_path=["kimm.models", "kimm.models.convnext"])
class ConvNeXtPico(ConvNeXtVariant):
    available_weights = [
        (
            "imagenet",
            ConvNeXt.default_origin,
            "convnextpico_convnext_pico.d1_in1k.keras",
        )
    ]

    # Parameters
    depths = (2, 2, 6, 2)
    hidden_channels = (64, 128, 256, 512)
    patch_size = 4
    kernel_size = 7
    activation = "gelu"
    use_conv_mlp = True


@kimm_export(parent_path=["kimm.models", "kimm.models.convnext"])
class ConvNeXtNano(ConvNeXtVariant):
    available_weights = [
        (
            "imagenet",
            ConvNeXt.default_origin,
            "convnextnano_convnext_nano.in12k_ft_in1k.keras",
        )
    ]

    # Parameters
    depths = (2, 2, 8, 2)
    hidden_channels = (80, 160, 320, 640)
    patch_size = 4
    kernel_size = 7
    activation = "gelu"
    use_conv_mlp = True


@kimm_export(parent_path=["kimm.models", "kimm.models.convnext"])
class ConvNeXtTiny(ConvNeXtVariant):
    available_weights = [
        (
            "imagenet",
            ConvNeXt.default_origin,
            "convnexttiny_convnext_tiny.in12k_ft_in1k.keras",
        )
    ]

    # Parameters
    depths = (3, 3, 9, 3)
    hidden_channels = (96, 192, 384, 768)
    patch_size = 4
    kernel_size = 7
    activation = "gelu"
    use_conv_mlp = False


@kimm_export(parent_path=["kimm.models", "kimm.models.convnext"])
class ConvNeXtSmall(ConvNeXtVariant):
    available_weights = [
        (
            "imagenet",
            ConvNeXt.default_origin,
            "convnextsmall_convnext_small.in12k_ft_in1k.keras",
        )
    ]

    # Parameters
    depths = (3, 3, 27, 3)
    hidden_channels = (96, 192, 384, 768)
    patch_size = 4
    kernel_size = 7
    activation = "gelu"
    use_conv_mlp = False


@kimm_export(parent_path=["kimm.models", "kimm.models.convnext"])
class ConvNeXtBase(ConvNeXtVariant):
    available_weights = [
        (
            "imagenet",
            ConvNeXt.default_origin,
            "convnextbase_convnext_base.fb_in22k_ft_in1k.keras",
        )
    ]

    # Parameters
    depths = (3, 3, 27, 3)
    hidden_channels = (128, 256, 512, 1024)
    patch_size = 4
    kernel_size = 7
    activation = "gelu"
    use_conv_mlp = False


@kimm_export(parent_path=["kimm.models", "kimm.models.convnext"])
class ConvNeXtLarge(ConvNeXtVariant):
    available_weights = [
        (
            "imagenet",
            ConvNeXt.default_origin,
            "convnextlarge_convnext_large.fb_in22k_ft_in1k.keras",
        )
    ]

    # Parameters
    depths = (3, 3, 27, 3)
    hidden_channels = (192, 384, 768, 1536)
    patch_size = 4
    kernel_size = 7
    activation = "gelu"
    use_conv_mlp = False


@kimm_export(parent_path=["kimm.models", "kimm.models.convnext"])
class ConvNeXtXLarge(ConvNeXtVariant):
    available_weights = []

    # Parameters
    depths = (3, 3, 27, 3)
    hidden_channels = (256, 512, 1024, 2048)
    patch_size = 4
    kernel_size = 7
    activation = "gelu"
    use_conv_mlp = False


add_model_to_registry(ConvNeXtAtto, "imagenet")
add_model_to_registry(ConvNeXtFemto, "imagenet")
add_model_to_registry(ConvNeXtPico, "imagenet")
add_model_to_registry(ConvNeXtNano, "imagenet")
add_model_to_registry(ConvNeXtTiny, "imagenet")
add_model_to_registry(ConvNeXtSmall, "imagenet")
add_model_to_registry(ConvNeXtBase, "imagenet")
add_model_to_registry(ConvNeXtLarge, "imagenet")
add_model_to_registry(ConvNeXtXLarge)

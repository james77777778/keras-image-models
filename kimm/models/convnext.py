import typing

import keras
from keras import initializers
from keras import layers

from kimm import layers as kimm_layers
from kimm.blocks import apply_mlp_block
from kimm.models.base_model import BaseModel
from kimm.utils import add_model_to_registry


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
    input_channels = inputs.shape[-1]
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
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm")(x)

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
    x = kimm_layers.LayerScale(
        output_channels, initializers.Constant(1e-6), name=f"{name}_layerscale"
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
    input_channels = inputs.shape[-1]
    x = inputs

    # Downsample
    if input_channels != output_channels or strides > 1:
        ds_ks = 2 if strides > 1 else 1
        x = layers.LayerNormalization(
            epsilon=1e-6, name=f"{name}_downsample_0"
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

        # Stem
        x = layers.Conv2D(
            hidden_channels[0],
            patch_size,
            patch_size,
            use_bias=True,
            name="stem_0_conv2d",
        )(x)
        x = layers.LayerNormalization(epsilon=1e-6, name="stem_1")(x)
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
        x = layers.LayerNormalization(epsilon=1e-6, name="head_norm")(x)
        x = layers.Dropout(rate=dropout_rate, name="head_dropout")(x)
        x = layers.Dense(
            classes, activation=classifier_activation, name="classifier"
        )(x)
        return x

    @staticmethod
    def available_feature_keys():
        feature_keys = ["STEM_S4"]
        feature_keys.extend([f"BLOCK{i}_S{2**(i+2)}" for i in range(4)])
        return feature_keys

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


"""
Model Definition
"""


class ConvNeXtAtto(ConvNeXt):
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
        weights: typing.Optional[str] = None,
        name: str = "ConvNeXtAtto",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            (2, 2, 6, 2),
            (40, 80, 160, 320),
            4,
            7,
            "gelu",
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
            **kwargs,
        )


class ConvNeXtFemto(ConvNeXt):
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
        weights: typing.Optional[str] = None,
        name: str = "ConvNeXtFemto",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            (2, 2, 6, 2),
            (48, 96, 192, 384),
            4,
            7,
            "gelu",
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
            **kwargs,
        )


class ConvNeXtPico(ConvNeXt):
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
        weights: typing.Optional[str] = None,
        name: str = "ConvNeXtPico",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            (2, 2, 6, 2),
            (64, 128, 256, 512),
            4,
            7,
            "gelu",
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
            **kwargs,
        )


class ConvNeXtNano(ConvNeXt):
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
        weights: typing.Optional[str] = None,
        name: str = "ConvNeXtNano",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            (2, 2, 8, 2),
            (80, 160, 320, 640),
            4,
            7,
            "gelu",
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
            **kwargs,
        )


class ConvNeXtTiny(ConvNeXt):
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
        weights: typing.Optional[str] = None,
        name: str = "ConvNeXtTiny",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            (3, 3, 9, 3),
            (96, 192, 384, 768),
            4,
            7,
            "gelu",
            False,
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


class ConvNeXtSmall(ConvNeXt):
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
        weights: typing.Optional[str] = None,
        name: str = "ConvNeXtSmall",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            (3, 3, 27, 3),
            (96, 192, 384, 768),
            4,
            7,
            "gelu",
            False,
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


class ConvNeXtBase(ConvNeXt):
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
        weights: typing.Optional[str] = None,
        name: str = "ConvNeXtBase",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            (3, 3, 27, 3),
            (128, 256, 512, 1024),
            4,
            7,
            "gelu",
            False,
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


class ConvNeXtLarge(ConvNeXt):
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
        weights: typing.Optional[str] = None,
        name: str = "ConvNeXtLarge",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            (3, 3, 27, 3),
            (192, 384, 768, 1536),
            4,
            7,
            "gelu",
            False,
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


class ConvNeXtXLarge(ConvNeXt):
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
        weights: typing.Optional[str] = None,
        name: str = "ConvNeXtXLarge",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            (3, 3, 27, 3),
            (256, 512, 1024, 2048),
            4,
            7,
            "gelu",
            False,
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


add_model_to_registry(ConvNeXtAtto, "imagenet")
add_model_to_registry(ConvNeXtFemto, "imagenet")
add_model_to_registry(ConvNeXtPico, "imagenet")
add_model_to_registry(ConvNeXtNano, "imagenet")
add_model_to_registry(ConvNeXtTiny, "imagenet")
add_model_to_registry(ConvNeXtSmall, "imagenet")
add_model_to_registry(ConvNeXtBase, "imagenet")
add_model_to_registry(ConvNeXtLarge, "imagenet")
add_model_to_registry(ConvNeXtXLarge, "imagenet")

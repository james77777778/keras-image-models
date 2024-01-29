import typing

import keras
from keras import backend

from kimm import layers as kimm_layers
from kimm.models.base_model import BaseModel
from kimm.utils import add_model_to_registry


@keras.saving.register_keras_serializable(package="kimm")
class RepVGG(BaseModel):
    available_feature_keys = [
        "STEM_S2",
        *[f"BLOCK{i}_S{j}" for i, j in zip(range(4), [4, 8, 16, 32])],
    ]

    def __init__(
        self,
        num_blocks: typing.Sequence[int],
        num_channels: typing.Sequence[int],
        stem_channels: int = 48,
        reparameterized: bool = False,
        **kwargs,
    ):
        kwargs["weights_url"] = self.get_weights_url(kwargs["weights"])
        if kwargs["weights_url"] is not None and reparameterized is True:
            raise ValueError(
                "Weights can only be loaded with `reparameterized=False`. "
                "You can first initialize the model with "
                "`reparameterized=False` then use "
                "`get_reparameterized_model` to get the converted model. "
                f"Received: weights={kwargs['weights']}, "
                f"reparameterized={reparameterized}"
            )

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

        # stem
        x = kimm_layers.RepConv2D(
            stem_channels,
            3,
            2,
            has_skip=False,
            reparameterized=reparameterized,
            activation="relu",
            name="stem",
        )(x)
        features["STEM_S2"] = x

        # stages
        current_strides = 2
        for current_stage_idx, (c, n) in enumerate(
            zip(num_channels, num_blocks)
        ):
            strides = 2
            current_strides *= strides
            # blocks
            for current_block_idx in range(n):
                strides = strides if current_block_idx == 0 else 1
                input_channels = x.shape[channels_axis]
                has_skip = input_channels == c and strides == 1
                name = f"stages_{current_stage_idx}_{current_block_idx}"
                x = kimm_layers.RepConv2D(
                    c,
                    3,
                    strides,
                    has_skip=has_skip,
                    reparameterized=reparameterized,
                    activation="relu",
                    name=name,
                )(x)

            # add feature
            features[f"BLOCK{current_stage_idx}_S{current_strides}"] = x

        # Head
        x = self.build_head(x)

        super().__init__(inputs=inputs, outputs=x, features=features, **kwargs)

        # All references to `self` below this line
        self.num_blocks = num_blocks
        self.num_channels = num_channels
        self.stem_channels = stem_channels
        self.reparameterized = reparameterized

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_blocks": self.num_blocks,
                "num_channels": self.num_channels,
                "stem_channels": self.stem_channels,
                "reparameterized": self.reparameterized,
            }
        )
        return config

    def fix_config(self, config):
        unused_kwargs = [
            "num_blocks",
            "num_channels",
            "stem_channels",
        ]
        for k in unused_kwargs:
            config.pop(k, None)
        return config

    def get_reparameterized_model(self):
        config = self.get_config()
        config["reparameterized"] = True
        config["weights"] = None
        model = RepVGG(**config)
        for layer, reparameterized_layer in zip(self.layers, model.layers):
            if hasattr(layer, "get_reparameterized_weights"):
                kernel, bias = layer.get_reparameterized_weights()
                reparameterized_layer.rep_conv2d.kernel.assign(kernel)
                reparameterized_layer.rep_conv2d.bias.assign(bias)
            else:
                for weight, target_weight in zip(
                    layer.weights, reparameterized_layer.weights
                ):
                    target_weight.assign(weight)
        return model


"""
Model Definition
"""


class RepVGGA0(RepVGG):
    available_weights = [
        (
            "imagenet",
            RepVGG.default_origin,
            "repvgga0_repvgg_a0.rvgg_in1k.keras",
        )
    ]

    def __init__(
        self,
        reparameterized: bool = False,
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        dropout_rate: float = 0.0,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = "imagenet",
        name: str = "RepVGGA0",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            [2, 4, 14, 1],
            [48, 96, 192, 1280],
            48,
            reparameterized=reparameterized,
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


class RepVGGA1(RepVGG):
    available_weights = [
        (
            "imagenet",
            RepVGG.default_origin,
            "repvgga1_repvgg_a1.rvgg_in1k.keras",
        )
    ]

    def __init__(
        self,
        reparameterized: bool = False,
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        dropout_rate: float = 0.0,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = "imagenet",
        name: str = "RepVGGA1",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            [2, 4, 14, 1],
            [64, 128, 256, 1280],
            64,
            reparameterized=reparameterized,
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


class RepVGGA2(RepVGG):
    available_weights = [
        (
            "imagenet",
            RepVGG.default_origin,
            "repvgga2_repvgg_a2.rvgg_in1k.keras",
        )
    ]

    def __init__(
        self,
        reparameterized: bool = False,
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        dropout_rate: float = 0.0,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = "imagenet",
        name: str = "RepVGGA2",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            [2, 4, 14, 1],
            [96, 192, 384, 1408],
            64,
            reparameterized=reparameterized,
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


class RepVGGB0(RepVGG):
    available_weights = [
        (
            "imagenet",
            RepVGG.default_origin,
            "repvggb0_repvgg_b0.rvgg_in1k.keras",
        )
    ]

    def __init__(
        self,
        reparameterized: bool = False,
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        dropout_rate: float = 0.0,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = "imagenet",
        name: str = "RepVGGB0",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            [4, 6, 16, 1],
            [64, 128, 256, 1280],
            64,
            reparameterized=reparameterized,
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


class RepVGGB1(RepVGG):
    available_weights = [
        (
            "imagenet",
            RepVGG.default_origin,
            "repvggb1_repvgg_b1.rvgg_in1k.keras",
        )
    ]

    def __init__(
        self,
        reparameterized: bool = False,
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        dropout_rate: float = 0.0,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = "imagenet",
        name: str = "RepVGGB1",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            [4, 6, 16, 1],
            [128, 256, 512, 2048],
            64,
            reparameterized=reparameterized,
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


class RepVGGB2(RepVGG):
    available_weights = [
        (
            "imagenet",
            RepVGG.default_origin,
            "repvggb2_repvgg_b2.rvgg_in1k.keras",
        )
    ]

    def __init__(
        self,
        reparameterized: bool = False,
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        dropout_rate: float = 0.0,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = "imagenet",
        name: str = "RepVGGB2",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            [4, 6, 16, 1],
            [160, 320, 640, 2560],
            64,
            reparameterized=reparameterized,
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


class RepVGGB3(RepVGG):
    available_weights = [
        (
            "imagenet",
            RepVGG.default_origin,
            "repvggb3_repvgg_b3.rvgg_in1k.keras",
        )
    ]

    def __init__(
        self,
        reparameterized: bool = False,
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        dropout_rate: float = 0.0,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = "imagenet",
        name: str = "RepVGGB3",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            [4, 6, 16, 1],
            [192, 384, 768, 2560],
            64,
            reparameterized=reparameterized,
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


add_model_to_registry(RepVGGA0, "imagenet")
add_model_to_registry(RepVGGA1, "imagenet")
add_model_to_registry(RepVGGA2, "imagenet")
add_model_to_registry(RepVGGB0, "imagenet")
add_model_to_registry(RepVGGB1, "imagenet")
add_model_to_registry(RepVGGB2, "imagenet")
add_model_to_registry(RepVGGB3, "imagenet")

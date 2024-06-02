import pathlib
import typing

import keras
from keras import backend

from kimm._src.kimm_export import kimm_export
from kimm._src.layers.reparameterizable_conv2d import ReparameterizableConv2D
from kimm._src.models.base_model import BaseModel
from kimm._src.utils.model_registry import add_model_to_registry


@keras.saving.register_keras_serializable(package="kimm")
class RepVGG(BaseModel):
    # Updated weights: use ReparameterizableConv2D
    default_origin = "https://github.com/james77777778/keras-image-models/releases/download/0.1.2/"
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
        input_tensor=None,
        **kwargs,
    ):
        if kwargs["weights"] is not None and reparameterized is True:
            raise ValueError(
                "Weights can only be loaded with `reparameterized=False`. "
                "You can first initialize the model with "
                "`reparameterized=False` then use "
                "`get_reparameterized_model` to get the converted model. "
                f"Received: weights={kwargs['weights']}, "
                f"reparameterized={reparameterized}"
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
        x = ReparameterizableConv2D(
            stem_channels,
            3,
            2,
            has_skip=False,
            branch_size=1,
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
                x = ReparameterizableConv2D(
                    c,
                    3,
                    strides,
                    has_skip=has_skip,
                    branch_size=1,
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
        for layer, rep_layer in zip(self.layers, model.layers):
            if hasattr(layer, "get_reparameterized_weights"):
                kernel, bias = layer.get_reparameterized_weights()
                rep_layer.reparameterized_conv2d.kernel.assign(kernel)
                rep_layer.reparameterized_conv2d.bias.assign(bias)
            else:
                for weight, target_weight in zip(
                    layer.weights, rep_layer.weights
                ):
                    target_weight.assign(weight)
        return model


# Model Definition


class RepVGGVariant(RepVGG):
    # Parameters
    num_blocks = None
    num_channels = None
    stem_channels = None

    def __init__(
        self,
        reparameterized: bool = False,
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
        """Instantiates the RepVGG architecture.

        Reference:
        - [RepVGG: Making VGG-style ConvNets Great Again(CVPR 2021)](https://arxiv.org/abs/2101.03697)

        Args:
            reparameterized: Whether to instantiate the model with
                reparameterized state. Defaults to `False`. Note that
                pretrained weights are only available with
                `reparameterized=False`.
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
        if type(self) is RepVGGVariant:
            raise NotImplementedError(
                f"Cannot instantiate base class: {self.__class__.__name__}. "
                "You should use its subclasses."
            )
        kwargs = self.fix_config(kwargs)
        super().__init__(
            num_blocks=self.num_blocks,
            num_channels=self.num_channels,
            stem_channels=self.stem_channels,
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
            name=name or str(self.__class__.__name__),
            feature_extractor=feature_extractor,
            feature_keys=feature_keys,
            **kwargs,
        )


@kimm_export(parent_path=["kimm.models", "kimm.models.repvgg"])
class RepVGGA0(RepVGGVariant):
    available_weights = [
        (
            "imagenet",
            RepVGG.default_origin,
            "repvgga0_repvgg_a0.rvgg_in1k.keras",
        )
    ]

    # Parameters
    num_blocks = [2, 4, 14, 1]
    num_channels = [48, 96, 192, 1280]
    stem_channels = 48


@kimm_export(parent_path=["kimm.models", "kimm.models.repvgg"])
class RepVGGA1(RepVGGVariant):
    available_weights = [
        (
            "imagenet",
            RepVGG.default_origin,
            "repvgga1_repvgg_a1.rvgg_in1k.keras",
        )
    ]

    # Parameters
    num_blocks = [2, 4, 14, 1]
    num_channels = [64, 128, 256, 1280]
    stem_channels = 64


@kimm_export(parent_path=["kimm.models", "kimm.models.repvgg"])
class RepVGGA2(RepVGGVariant):
    available_weights = [
        (
            "imagenet",
            RepVGG.default_origin,
            "repvgga2_repvgg_a2.rvgg_in1k.keras",
        )
    ]

    # Parameters
    num_blocks = [2, 4, 14, 1]
    num_channels = [96, 192, 384, 1408]
    stem_channels = 64


@kimm_export(parent_path=["kimm.models", "kimm.models.repvgg"])
class RepVGGB0(RepVGGVariant):
    available_weights = [
        (
            "imagenet",
            RepVGG.default_origin,
            "repvggb0_repvgg_b0.rvgg_in1k.keras",
        )
    ]

    # Parameters
    num_blocks = [4, 6, 16, 1]
    num_channels = [64, 128, 256, 1280]
    stem_channels = 64


@kimm_export(parent_path=["kimm.models", "kimm.models.repvgg"])
class RepVGGB1(RepVGGVariant):
    available_weights = [
        (
            "imagenet",
            RepVGG.default_origin,
            "repvggb1_repvgg_b1.rvgg_in1k.keras",
        )
    ]

    # Parameters
    num_blocks = [4, 6, 16, 1]
    num_channels = [128, 256, 512, 2048]
    stem_channels = 64


@kimm_export(parent_path=["kimm.models", "kimm.models.repvgg"])
class RepVGGB2(RepVGGVariant):
    available_weights = [
        (
            "imagenet",
            RepVGG.default_origin,
            "repvggb2_repvgg_b2.rvgg_in1k.keras",
        )
    ]

    # Parameters
    num_blocks = [4, 6, 16, 1]
    num_channels = [160, 320, 640, 2560]
    stem_channels = 64


@kimm_export(parent_path=["kimm.models", "kimm.models.repvgg"])
class RepVGGB3(RepVGGVariant):
    available_weights = [
        (
            "imagenet",
            RepVGG.default_origin,
            "repvggb3_repvgg_b3.rvgg_in1k.keras",
        )
    ]

    # Parameters
    num_blocks = [4, 6, 16, 1]
    num_channels = [192, 384, 768, 2560]
    stem_channels = 64


add_model_to_registry(RepVGGA0, "imagenet")
add_model_to_registry(RepVGGA1, "imagenet")
add_model_to_registry(RepVGGA2, "imagenet")
add_model_to_registry(RepVGGB0, "imagenet")
add_model_to_registry(RepVGGB1, "imagenet")
add_model_to_registry(RepVGGB2, "imagenet")
add_model_to_registry(RepVGGB3, "imagenet")

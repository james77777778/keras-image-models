import typing

import keras
from keras import backend

from kimm._src.kimm_export import kimm_export
from kimm._src.layers.mobile_one_conv2d import MobileOneConv2D
from kimm._src.models.base_model import BaseModel
from kimm._src.utils.model_registry import add_model_to_registry


@keras.saving.register_keras_serializable(package="kimm")
class MobileOne(BaseModel):
    available_feature_keys = [
        "STEM_S2",
        *[f"BLOCK{i}_S{j}" for i, j in zip(range(4), [4, 8, 16, 32])],
    ]

    def __init__(
        self,
        num_blocks: typing.Sequence[int],
        num_channels: typing.Sequence[int],
        stem_channels: int = 48,
        branch_size: int = 1,
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
        x = MobileOneConv2D(
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
            current_block_idx = 0
            # blocks
            for _ in range(n):
                strides = strides if current_block_idx == 0 else 1
                input_channels = x.shape[channels_axis]
                has_skip1 = strides == 1
                has_skip2 = input_channels == c
                name1 = f"stages_{current_stage_idx}_{current_block_idx}"
                name2 = f"stages_{current_stage_idx}_{current_block_idx+1}"
                # Depthwise
                x = MobileOneConv2D(
                    input_channels,
                    3,
                    strides,
                    has_skip=has_skip1,
                    use_depthwise=True,
                    branch_size=branch_size,
                    reparameterized=reparameterized,
                    activation="relu",
                    name=name1,
                )(x)
                # Pointwise
                x = MobileOneConv2D(
                    c,
                    1,
                    1,
                    has_skip=has_skip2,
                    use_depthwise=False,
                    branch_size=branch_size,
                    reparameterized=reparameterized,
                    activation="relu",
                    name=name2,
                )(x)
                current_block_idx += 2

            # add feature
            features[f"BLOCK{current_stage_idx}_S{current_strides}"] = x

        # Head
        x = self.build_head(x)

        super().__init__(inputs=inputs, outputs=x, features=features, **kwargs)

        # All references to `self` below this line
        self.num_blocks = num_blocks
        self.num_channels = num_channels
        self.stem_channels = stem_channels
        self.branch_size = branch_size
        self.reparameterized = reparameterized

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_blocks": self.num_blocks,
                "num_channels": self.num_channels,
                "stem_channels": self.stem_channels,
                "branch_size": self.branch_size,
                "reparameterized": self.reparameterized,
            }
        )
        return config

    def fix_config(self, config):
        unused_kwargs = [
            "num_blocks",
            "num_channels",
            "stem_channels",
            "branch_size",
        ]
        for k in unused_kwargs:
            config.pop(k, None)
        return config

    def get_reparameterized_model(self):
        config = self.get_config()
        config["reparameterized"] = True
        config["weights"] = None
        model = MobileOne(**config)
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


# Model Definition


class MobileOneVariant(MobileOne):
    # Parameters
    num_blocks = None
    num_channels = None
    stem_channels = None
    branch_size = None

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
        name: typing.Optional[str] = None,
        **kwargs,
    ):
        if type(self) is MobileOneVariant:
            raise NotImplementedError(
                f"Cannot instantiate base class: {self.__class__.__name__}. "
                "You should use its subclasses."
            )
        kwargs = self.fix_config(kwargs)
        super().__init__(
            num_blocks=self.num_blocks,
            num_channels=self.num_channels,
            stem_channels=self.stem_channels,
            branch_size=self.branch_size,
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
            **kwargs,
        )


@kimm_export(parent_path=["kimm.models", "kimm.models.mobileone"])
class MobileOneS0(MobileOneVariant):
    available_weights = [
        (
            "imagenet",
            MobileOne.default_origin,
            "mobileones0_mobileone_s0.apple_in1k.keras",
        )
    ]

    # Parameters
    num_blocks = [2, 8, 10, 1]
    num_channels = [48, 128, 256, 1024]
    stem_channels = 48
    branch_size = 4


@kimm_export(parent_path=["kimm.models", "kimm.models.mobileone"])
class MobileOneS1(MobileOneVariant):
    available_weights = [
        (
            "imagenet",
            MobileOne.default_origin,
            "mobileones1_mobileone_s1.apple_in1k.keras",
        )
    ]

    # Parameters
    num_blocks = [2, 8, 10, 1]
    num_channels = [96, 192, 512, 1280]
    stem_channels = 64
    branch_size = 1


@kimm_export(parent_path=["kimm.models", "kimm.models.mobileone"])
class MobileOneS2(MobileOneVariant):
    available_weights = [
        (
            "imagenet",
            MobileOne.default_origin,
            "mobileones2_mobileone_s2.apple_in1k.keras",
        )
    ]

    # Parameters
    num_blocks = [2, 8, 10, 1]
    num_channels = [96, 256, 640, 2048]
    stem_channels = 64
    branch_size = 1


@kimm_export(parent_path=["kimm.models", "kimm.models.mobileone"])
class MobileOneS3(MobileOneVariant):
    available_weights = [
        (
            "imagenet",
            MobileOne.default_origin,
            "mobileones3_mobileone_s3.apple_in1k.keras",
        )
    ]

    # Parameters
    num_blocks = [2, 8, 10, 1]
    num_channels = [128, 320, 768, 2048]
    stem_channels = 64
    branch_size = 1


# TODO: Add MobileOneS4 (w/ SE blocks)


add_model_to_registry(MobileOneS0, "imagenet")
add_model_to_registry(MobileOneS1, "imagenet")
add_model_to_registry(MobileOneS2, "imagenet")
add_model_to_registry(MobileOneS3, "imagenet")

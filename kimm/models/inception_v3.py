import functools
import typing

import keras
from keras import backend
from keras import layers

from kimm.blocks import apply_conv2d_block
from kimm.models import BaseModel
from kimm.utils import add_model_to_registry

_apply_conv2d_block = functools.partial(
    apply_conv2d_block, activation="relu", bn_epsilon=1e-3, padding="valid"
)


def apply_inception_a_block(inputs, pool_channels, name="inception_a_block"):
    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3

    x = inputs

    branch1x1 = _apply_conv2d_block(x, 64, 1, 1, name=f"{name}_branch1x1")

    branch5x5 = _apply_conv2d_block(x, 48, 1, 1, name=f"{name}_branch5x5_1")
    branch5x5 = _apply_conv2d_block(
        branch5x5, 64, 5, 1, padding=None, name=f"{name}_branch5x5_2"
    )

    branch3x3dbl = _apply_conv2d_block(
        x, 64, 1, 1, name=f"{name}_branch3x3dbl_1"
    )
    branch3x3dbl = _apply_conv2d_block(
        branch3x3dbl, 96, 3, 1, padding=None, name=f"{name}_branch3x3dbl_2"
    )
    branch3x3dbl = _apply_conv2d_block(
        branch3x3dbl, 96, 3, 1, padding=None, name=f"{name}_branch3x3dbl_3"
    )

    branch_pool = layers.ZeroPadding2D(1)(x)
    branch_pool = layers.AveragePooling2D(
        3, 1, data_format=backend.image_data_format()
    )(branch_pool)
    branch_pool = _apply_conv2d_block(
        branch_pool,
        pool_channels,
        1,
        1,
        activation="relu",
        name=f"{name}_branch_pool",
    )
    x = layers.Concatenate(axis=channels_axis)(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool]
    )
    return x


def apply_inception_b_block(inputs, name="incpetion_b_block"):
    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3

    x = inputs

    branch3x3 = _apply_conv2d_block(x, 384, 3, 2, name=f"{name}_branch3x3")

    branch3x3dbl = _apply_conv2d_block(
        x, 64, 1, 1, name=f"{name}_branch3x3dbl_1"
    )
    branch3x3dbl = _apply_conv2d_block(
        branch3x3dbl, 96, 3, 1, padding=None, name=f"{name}_branch3x3dbl_2"
    )
    branch3x3dbl = _apply_conv2d_block(
        branch3x3dbl, 96, 3, 2, name=f"{name}_branch3x3dbl_3"
    )

    branch_pool = layers.MaxPooling2D(3, 2, name=f"{name}_branch_pool")(x)
    x = layers.Concatenate(axis=channels_axis)(
        [branch3x3, branch3x3dbl, branch_pool]
    )
    return x


def apply_inception_c_block(
    inputs, branch7x7_channels, name="inception_c_block"
):
    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3
    c7 = branch7x7_channels

    x = inputs

    branch1x1 = _apply_conv2d_block(x, 192, 1, 1, name=f"{name}_branch1x1")

    branch7x7 = _apply_conv2d_block(x, c7, 1, 1, name=f"{name}_branch7x7_1")
    branch7x7 = _apply_conv2d_block(
        branch7x7, c7, (1, 7), 1, padding=None, name=f"{name}_branch7x7_2"
    )
    branch7x7 = _apply_conv2d_block(
        branch7x7, 192, (7, 1), 1, padding=None, name=f"{name}_branch7x7_3"
    )

    branch7x7dbl = _apply_conv2d_block(
        x, c7, 1, 1, name=f"{name}_branch7x7dbl_1"
    )
    branch7x7dbl = _apply_conv2d_block(
        branch7x7dbl, c7, (7, 1), 1, padding=None, name=f"{name}_branch7x7dbl_2"
    )
    branch7x7dbl = _apply_conv2d_block(
        branch7x7dbl, c7, (1, 7), 1, padding=None, name=f"{name}_branch7x7dbl_3"
    )
    branch7x7dbl = _apply_conv2d_block(
        branch7x7dbl, c7, (7, 1), 1, padding=None, name=f"{name}_branch7x7dbl_4"
    )
    branch7x7dbl = _apply_conv2d_block(
        branch7x7dbl,
        192,
        (1, 7),
        1,
        padding=None,
        name=f"{name}_branch7x7dbl_5",
    )

    branch_pool = layers.ZeroPadding2D(1)(x)
    branch_pool = layers.AveragePooling2D(
        3, 1, data_format=backend.image_data_format()
    )(branch_pool)
    branch_pool = _apply_conv2d_block(
        branch_pool, 192, 1, 1, name=f"{name}_branch_pool"
    )
    x = layers.Concatenate(axis=channels_axis)(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool]
    )
    return x


def apply_inception_d_block(inputs, name="inception_d_block"):
    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3

    x = inputs

    branch3x3 = _apply_conv2d_block(x, 192, 1, 1, name=f"{name}_branch3x3_1")
    branch3x3 = _apply_conv2d_block(
        branch3x3, 320, 3, 2, name=f"{name}_branch3x3_2"
    )

    branch7x7x3 = _apply_conv2d_block(
        x, 192, 1, 1, name=f"{name}_branch7x7x3_1"
    )
    branch7x7x3 = _apply_conv2d_block(
        branch7x7x3, 192, (1, 7), 1, padding=None, name=f"{name}_branch7x7x3_2"
    )
    branch7x7x3 = _apply_conv2d_block(
        branch7x7x3, 192, (7, 1), 1, padding=None, name=f"{name}_branch7x7x3_3"
    )
    branch7x7x3 = _apply_conv2d_block(
        branch7x7x3, 192, 3, 2, name=f"{name}_branch7x7x3_4"
    )

    branch_pool = layers.MaxPooling2D(3, 2)(x)
    x = layers.Concatenate(axis=channels_axis)(
        [branch3x3, branch7x7x3, branch_pool]
    )
    return x


def apply_inception_e_block(inputs, name="inception_e_block"):
    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3

    x = inputs

    branch1x1 = _apply_conv2d_block(x, 320, 1, 1, name=f"{name}_branch1x1")

    branch3x3 = _apply_conv2d_block(x, 384, 1, 1, name=f"{name}_branch3x3_1")
    branch3x3 = [
        _apply_conv2d_block(
            branch3x3, 384, (1, 3), 1, padding=None, name=f"{name}_branch3x3_2a"
        ),
        _apply_conv2d_block(
            branch3x3, 384, (3, 1), 1, padding=None, name=f"{name}_branch3x3_2b"
        ),
    ]
    branch3x3 = layers.Concatenate(axis=channels_axis)(branch3x3)

    branch3x3dbl = _apply_conv2d_block(
        x, 448, 1, 1, name=f"{name}_branch3x3dbl_1"
    )
    branch3x3dbl = _apply_conv2d_block(
        branch3x3dbl, 384, 3, 1, padding=None, name=f"{name}_branch3x3dbl_2"
    )
    branch3x3dbl = [
        _apply_conv2d_block(
            branch3x3dbl,
            384,
            (1, 3),
            1,
            padding=None,
            name=f"{name}_branch3x3dbl_3a",
        ),
        _apply_conv2d_block(
            branch3x3dbl,
            384,
            (3, 1),
            1,
            padding=None,
            name=f"{name}_branch3x3dbl_3b",
        ),
    ]
    branch3x3dbl = layers.Concatenate(axis=channels_axis)(branch3x3dbl)

    branch_pool = layers.ZeroPadding2D(1)(x)
    branch_pool = layers.AveragePooling2D(
        3, 1, data_format=backend.image_data_format()
    )(branch_pool)
    branch_pool = _apply_conv2d_block(
        branch_pool, 192, 1, 1, name=f"{name}_branch_pool"
    )
    x = layers.Concatenate(axis=channels_axis)(
        [branch1x1, branch3x3, branch3x3dbl, branch_pool]
    )
    return x


def apply_inception_aux_block(inputs, classes, name="inception_aux_block"):
    x = inputs

    x = layers.AveragePooling2D(5, 3, data_format=backend.image_data_format())(
        x
    )
    x = _apply_conv2d_block(x, 128, 1, 1, name=f"{name}_conv0")
    x = _apply_conv2d_block(x, 768, 5, 1, name=f"{name}_conv1")
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(classes, use_bias=True, name=f"{name}_fc")(x)
    return x


@keras.saving.register_keras_serializable(package="kimm")
class InceptionV3Base(BaseModel):
    available_feature_keys = [
        "STEM_S2",
        *[f"BLOCK{i}_S{j}" for i, j in zip(range(4), [4, 8, 16, 32])],
    ]

    def __init__(self, has_aux_logits: bool = False, **kwargs):
        kwargs["weights_url"] = self.get_weights_url(kwargs["weights"])

        input_tensor = kwargs.pop("input_tensor", None)
        self.set_properties(kwargs, 299)
        inputs = self.determine_input_tensor(
            input_tensor,
            self._input_shape,
            self._default_size,
            require_flatten=self._include_top,
        )
        x = inputs

        x = self.build_preprocessing(x, "imagenet")

        # Prepare feature extraction
        features = {}

        # Stem block
        x = _apply_conv2d_block(x, 32, 3, 2, name="Conv2d_1a_3x3")
        x = _apply_conv2d_block(x, 32, 3, 1, name="Conv2d_2a_3x3")
        x = _apply_conv2d_block(x, 64, 3, 1, padding=None, name="Conv2d_2b_3x3")
        features["STEM_S2"] = x

        # Blocks
        x = layers.MaxPooling2D(3, 2, name="Pool1")(x)
        x = _apply_conv2d_block(x, 80, 1, 1, name="Conv2d_3b_1x1")
        x = _apply_conv2d_block(x, 192, 3, 1, name="Conv2d_4a_3x3")
        features["BLOCK0_S4"] = x
        x = layers.MaxPooling2D(3, 2, name="Pool2")(x)
        x = apply_inception_a_block(x, 32, "Mixed_5b")
        x = apply_inception_a_block(x, 64, "Mixed_5c")
        x = apply_inception_a_block(x, 64, "Mixed_5d")
        features["BLOCK1_S8"] = x

        x = apply_inception_b_block(x, "Mixed_6a")

        x = apply_inception_c_block(x, 128, "Mixed_6b")
        x = apply_inception_c_block(x, 160, "Mixed_6c")
        x = apply_inception_c_block(x, 160, "Mixed_6d")
        x = apply_inception_c_block(x, 192, "Mixed_6e")
        features["BLOCK2_S16"] = x

        if has_aux_logits:
            aux_logits = apply_inception_aux_block(
                x, self._classes, "AuxLogits"
            )

        x = apply_inception_d_block(x, "Mixed_7a")
        x = apply_inception_e_block(x, "Mixed_7b")
        x = apply_inception_e_block(x, "Mixed_7c")
        features["BLOCK3_S32"] = x

        # Head
        x = self.build_head(x)

        if has_aux_logits:
            x = [x, aux_logits]
        super().__init__(inputs=inputs, outputs=x, features=features, **kwargs)

        # All references to `self` below this line
        self.has_aux_logits = has_aux_logits

    def get_config(self):
        config = super().get_config()
        config.update({"has_aux_logits": self.has_aux_logits})
        return config

    def fix_config(self, config: typing.Dict):
        return config


class InceptionV3(InceptionV3Base):
    available_weights = [
        (
            "imagenet_aux_logits",
            InceptionV3Base.default_origin,
            "inceptionv3_inception_v3.gluon_in1k_aux_logits.keras",
        ),
        (
            "imagenet_no_aux_logits",
            InceptionV3Base.default_origin,
            "inceptionv3_inception_v3.gluon_in1k_no_aux_logits.keras",
        ),
    ]

    def __init__(
        self,
        has_aux_logits: bool = False,
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        dropout_rate: float = 0.0,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = "imagenet",
        name: str = "InceptionV3",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        if weights == "imagenet":
            if has_aux_logits:
                weights = f"{weights}_aux_logits"
            else:
                weights = f"{weights}_no_aux_logits"
        super().__init__(
            has_aux_logits,
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


add_model_to_registry(InceptionV3, "imagenet")

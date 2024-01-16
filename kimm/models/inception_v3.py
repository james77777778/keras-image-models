import functools
import typing

import keras
from keras import backend
from keras import layers
from keras import utils
from keras.src.applications import imagenet_utils

from kimm.blocks import apply_conv2d_block
from kimm.models.feature_extractor import FeatureExtractor
from kimm.utils import add_model_to_registry

_apply_conv2d_block = functools.partial(
    apply_conv2d_block, activation="relu", bn_epsilon=1e-3, padding="valid"
)


def apply_inception_a_block(inputs, pool_channels, name="inception_a_block"):
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
    branch_pool = layers.AveragePooling2D(3, 1)(branch_pool)
    branch_pool = _apply_conv2d_block(
        branch_pool,
        pool_channels,
        1,
        1,
        activation="relu",
        name=f"{name}_branch_pool",
    )
    x = layers.Concatenate()([branch1x1, branch5x5, branch3x3dbl, branch_pool])
    return x


def apply_inception_b_block(inputs, name="incpetion_b_block"):
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
    x = layers.Concatenate()([branch3x3, branch3x3dbl, branch_pool])
    return x


def apply_inception_c_block(
    inputs, branch7x7_channels, name="inception_c_block"
):
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
    branch_pool = layers.AveragePooling2D(3, 1)(branch_pool)
    branch_pool = _apply_conv2d_block(
        branch_pool, 192, 1, 1, name=f"{name}_branch_pool"
    )
    x = layers.Concatenate()([branch1x1, branch7x7, branch7x7dbl, branch_pool])
    return x


def apply_inception_d_block(inputs, name="inception_d_block"):
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
    x = layers.Concatenate()([branch3x3, branch7x7x3, branch_pool])
    return x


def apply_inception_e_block(inputs, name="inception_e_block"):
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
    branch3x3 = layers.Concatenate()(branch3x3)

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
    branch3x3dbl = layers.Concatenate()(branch3x3dbl)

    branch_pool = layers.ZeroPadding2D(1)(x)
    branch_pool = layers.AveragePooling2D(3, 1)(branch_pool)
    branch_pool = _apply_conv2d_block(
        branch_pool, 192, 1, 1, name=f"{name}_branch_pool"
    )
    x = layers.Concatenate()([branch1x1, branch3x3, branch3x3dbl, branch_pool])
    return x


def apply_inception_aux_block(inputs, classes, name="inception_aux_block"):
    x = inputs

    x = layers.AveragePooling2D(5, 3)(x)
    x = _apply_conv2d_block(x, 128, 1, 1, name=f"{name}_conv0")
    x = _apply_conv2d_block(x, 768, 5, 1, name=f"{name}_conv1")
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(classes, use_bias=True, name=f"{name}_fc")(x)
    return x


class InceptionV3Base(FeatureExtractor):
    def __init__(
        self,
        has_aux_logits=False,
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        dropout_rate: float = 0.0,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = None,  # TODO: imagenet
        **kwargs,
    ):
        # default_size
        default_size = kwargs.pop("default_size", 299)

        # Prepare feature extraction
        features = {}

        # Determine proper input shape
        input_shape = imagenet_utils.obtain_input_shape(
            input_shape,
            default_size=default_size,
            min_size=32,
            data_format=backend.image_data_format(),
            require_flatten=include_top,
            weights=weights,
        )

        if input_tensor is None:
            img_input = layers.Input(shape=input_shape)
        else:
            if not backend.is_keras_tensor(input_tensor):
                img_input = layers.Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        x = img_input

        # [0, 255] to [0, 1] and apply ImageNet mean and variance
        if include_preprocessing:
            x = layers.Rescaling(scale=1.0 / 255.0)(x)
            x = layers.Normalization(
                mean=[0.485, 0.456, 0.406], variance=[0.229, 0.224, 0.225]
            )(x)

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
            aux_logits = apply_inception_aux_block(x, classes, "AuxLogits")

        x = apply_inception_d_block(x, "Mixed_7a")
        x = apply_inception_e_block(x, "Mixed_7b")
        x = apply_inception_e_block(x, "Mixed_7c")
        features["BLOCK3_S32"] = x

        # Head
        if include_top:
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            x = layers.Dropout(rate=dropout_rate, name="head_dropout")(x)
            x = layers.Dense(
                classes, activation=classifier_activation, name="classifier"
            )(x)
        else:
            if pooling == "avg":
                x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            elif pooling == "max":
                x = layers.GlobalMaxPooling2D(name="max_pool")(x)

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        if input_tensor is not None:
            inputs = utils.get_source_inputs(input_tensor)
        else:
            inputs = img_input

        if has_aux_logits:
            x = [x, aux_logits]
        super().__init__(inputs=inputs, outputs=x, features=features, **kwargs)

        # All references to `self` below this line
        self.has_aux_logits = has_aux_logits
        self.include_preprocessing = include_preprocessing
        self.include_top = include_top
        self.pooling = pooling
        self.dropout_rate = dropout_rate
        self.classes = classes
        self.classifier_activation = classifier_activation
        self._weights = weights  # `self.weights` is been used internally

    @staticmethod
    def available_feature_keys():
        feature_keys = ["STEM_S2"]
        feature_keys.extend(
            [f"BLOCK{i}_S{j}" for i, j in zip(range(4), [4, 8, 16, 32])]
        )
        return feature_keys

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "has_aux_logits": self.has_aux_logits,
                "input_shape": self.input_shape[1:],
                "include_preprocessing": self.include_preprocessing,
                "include_top": self.include_top,
                "pooling": self.pooling,
                "dropout_rate": self.dropout_rate,
                "classes": self.classes,
                "classifier_activation": self.classifier_activation,
                "weights": self._weights,
            }
        )
        return config

    def fix_config(self, config: typing.Dict):
        return config


class InceptionV3(InceptionV3Base):
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
        weights: typing.Optional[str] = None,  # TODO: imagenet
        name: str = "InceptionV3",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            has_aux_logits,
            input_tensor,
            input_shape,
            include_preprocessing,
            include_top,
            pooling,
            dropout_rate,
            classes,
            classifier_activation,
            weights,
            name=name,
            **kwargs,
        )


add_model_to_registry(InceptionV3, True)

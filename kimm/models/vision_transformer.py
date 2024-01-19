import typing

import keras
from keras import layers

from kimm import layers as kimm_layers
from kimm.blocks import apply_transformer_block
from kimm.models.base_model import BaseModel
from kimm.utils import add_model_to_registry


@keras.saving.register_keras_serializable(package="kimm")
class VisionTransformer(BaseModel):
    def __init__(
        self,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_qkv_bias: bool = True,
        use_qk_norm: bool = False,
        pos_dropout_rate: float = 0.0,
        **kwargs,
    ):
        input_tensor = kwargs.pop("input_tensor", None)
        self.set_properties(kwargs, 384)
        if self._pooling is not None:
            raise ValueError(
                "`VisionTransformer` doesn't support `pooling`. "
                f"Received: pooling={self._pooling}"
            )
        inputs = self.determine_input_tensor(
            input_tensor,
            self._input_shape,
            self._default_size,
            static_shape=True,
        )
        x = inputs

        x = self.build_preprocessing(x, "-1_1")

        # Prepare feature extraction
        features = {}

        # patch embedding
        x = layers.Conv2D(
            embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            use_bias=True,
            name="patch_embed_conv",
        )(x)
        x = layers.Reshape((-1, embed_dim))(x)
        x = kimm_layers.PositionEmbedding(name="postition_embedding")(x)
        features["EMBEDDING"] = x
        x = layers.Dropout(pos_dropout_rate, name="pos_dropout")(x)

        for i in range(depth):
            x = apply_transformer_block(
                x,
                embed_dim,
                num_heads,
                mlp_ratio,
                use_qkv_bias,
                use_qk_norm,
                activation="gelu",
                name=f"blocks_{i}",
            )
            features[f"BLOCK{i}"] = x
        x = layers.LayerNormalization(epsilon=1e-6, name="norm")(x)

        # Head
        if self._include_top:
            x = self.build_top(
                x,
                self._classes,
                self._classifier_activation,
                self._dropout_rate,
            )

        super().__init__(inputs=inputs, outputs=x, features=features, **kwargs)

        # All references to `self` below this line
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.use_qkv_bias = use_qkv_bias
        self.use_qk_norm = use_qk_norm
        self.pos_dropout_rate = pos_dropout_rate

    def build_top(self, inputs, classes, classifier_activation, dropout_rate):
        x = inputs[:, 0]  # class token
        x = layers.Dropout(dropout_rate, name="head_drop")(x)
        x = layers.Dense(
            classes, activation=classifier_activation, name="head"
        )(x)
        return x

    @staticmethod
    def available_feature_keys():
        raise NotImplementedError()

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "embed_dim": self.embed_dim,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "use_qkv_bias": self.use_qkv_bias,
                "use_qk_norm": self.use_qk_norm,
                "pos_dropout_rate": self.pos_dropout_rate,
            }
        )
        return config

    def fix_config(self, config):
        unused_kwargs = [
            "patch_size",
            "embed_dim",
            "depth",
            "num_heads",
            "mlp_ratio",
            "use_qkv_bias",
            "use_qk_norm",
            "pos_dropout_rate",
        ]
        for k in unused_kwargs:
            config.pop(k, None)
        return config


"""
Model Definition
"""


class VisionTransformerTiny16(VisionTransformer):
    def __init__(
        self,
        mlp_ratio: float = 4.0,
        use_qkv_bias: bool = True,
        use_qk_norm: bool = False,
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        pos_dropout_rate: float = 0.0,
        dropout_rate: float = 0.1,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = None,  # TODO: imagenet
        name: str = "VisionTransformerTiny16",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            16,
            192,
            12,
            3,
            mlp_ratio,
            use_qkv_bias,
            use_qk_norm,
            pos_dropout_rate,
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

    @staticmethod
    def available_feature_keys():
        feature_keys = ["EMBEDDING"]
        feature_keys.extend([f"BLOCK{i}" for i in range(12)])
        return feature_keys


class VisionTransformerTiny32(VisionTransformer):
    def __init__(
        self,
        mlp_ratio: float = 4.0,
        use_qkv_bias: bool = True,
        use_qk_norm: bool = False,
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        pos_dropout_rate: float = 0.0,
        dropout_rate: float = 0.1,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = None,  # TODO: imagenet
        name: str = "VisionTransformerTiny32",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            32,
            192,
            12,
            3,
            mlp_ratio,
            use_qkv_bias,
            use_qk_norm,
            pos_dropout_rate,
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

    @staticmethod
    def available_feature_keys():
        feature_keys = ["EMBEDDING"]
        feature_keys.extend([f"BLOCK{i}" for i in range(12)])
        return feature_keys


class VisionTransformerSmall16(VisionTransformer):
    def __init__(
        self,
        mlp_ratio: float = 4.0,
        use_qkv_bias: bool = True,
        use_qk_norm: bool = False,
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        pos_dropout_rate: float = 0.0,
        dropout_rate: float = 0.1,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = None,  # TODO: imagenet
        name: str = "VisionTransformerSmall16",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            16,
            384,
            12,
            6,
            mlp_ratio,
            use_qkv_bias,
            use_qk_norm,
            pos_dropout_rate,
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

    @staticmethod
    def available_feature_keys():
        feature_keys = ["EMBEDDING"]
        feature_keys.extend([f"BLOCK{i}" for i in range(12)])
        return feature_keys


class VisionTransformerSmall32(VisionTransformer):
    def __init__(
        self,
        mlp_ratio: float = 4.0,
        use_qkv_bias: bool = True,
        use_qk_norm: bool = False,
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        pos_dropout_rate: float = 0.0,
        dropout_rate: float = 0.1,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = None,  # TODO: imagenet
        name: str = "VisionTransformerSmall32",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            32,
            384,
            12,
            6,
            mlp_ratio,
            use_qkv_bias,
            use_qk_norm,
            pos_dropout_rate,
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

    @staticmethod
    def available_feature_keys():
        feature_keys = ["EMBEDDING"]
        feature_keys.extend([f"BLOCK{i}" for i in range(12)])
        return feature_keys


class VisionTransformerBase16(VisionTransformer):
    def __init__(
        self,
        mlp_ratio: float = 4.0,
        use_qkv_bias: bool = True,
        use_qk_norm: bool = False,
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        pos_dropout_rate: float = 0.0,
        dropout_rate: float = 0.1,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = None,  # TODO: imagenet
        name: str = "VisionTransformerBase16",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            16,
            768,
            12,
            12,
            mlp_ratio,
            use_qkv_bias,
            use_qk_norm,
            pos_dropout_rate,
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

    @staticmethod
    def available_feature_keys():
        feature_keys = ["EMBEDDING"]
        feature_keys.extend([f"BLOCK{i}" for i in range(12)])
        return feature_keys


class VisionTransformerBase32(VisionTransformer):
    def __init__(
        self,
        mlp_ratio: float = 4.0,
        use_qkv_bias: bool = True,
        use_qk_norm: bool = False,
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        pos_dropout_rate: float = 0.0,
        dropout_rate: float = 0.1,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = None,  # TODO: imagenet
        name: str = "VisionTransformerBase32",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            32,
            768,
            12,
            12,
            mlp_ratio,
            use_qkv_bias,
            use_qk_norm,
            pos_dropout_rate,
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

    @staticmethod
    def available_feature_keys():
        feature_keys = ["EMBEDDING"]
        feature_keys.extend([f"BLOCK{i}" for i in range(12)])
        return feature_keys


class VisionTransformerLarge16(VisionTransformer):
    def __init__(
        self,
        mlp_ratio: float = 4.0,
        use_qkv_bias: bool = True,
        use_qk_norm: bool = False,
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        pos_dropout_rate: float = 0.0,
        dropout_rate: float = 0.1,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = None,  # TODO: imagenet
        name: str = "VisionTransformerLarge16",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            16,
            1024,
            24,
            16,
            mlp_ratio,
            use_qkv_bias,
            use_qk_norm,
            pos_dropout_rate,
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

    @staticmethod
    def available_feature_keys():
        feature_keys = ["EMBEDDING"]
        feature_keys.extend([f"BLOCK{i}" for i in range(24)])
        return feature_keys


class VisionTransformerLarge32(VisionTransformer):
    def __init__(
        self,
        mlp_ratio: float = 4.0,
        use_qkv_bias: bool = True,
        use_qk_norm: bool = False,
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        pos_dropout_rate: float = 0.0,
        dropout_rate: float = 0.1,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = None,  # TODO: imagenet
        name: str = "VisionTransformerLarge32",
        **kwargs,
    ):
        kwargs = self.fix_config(kwargs)
        super().__init__(
            32,
            1024,
            24,
            16,
            mlp_ratio,
            use_qkv_bias,
            use_qk_norm,
            pos_dropout_rate,
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

    @staticmethod
    def available_feature_keys():
        feature_keys = ["EMBEDDING"]
        feature_keys.extend([f"BLOCK{i}" for i in range(24)])
        return feature_keys


add_model_to_registry(VisionTransformerTiny16, "imagenet")
add_model_to_registry(VisionTransformerTiny32, "imagenet")
add_model_to_registry(VisionTransformerSmall16, "imagenet")
add_model_to_registry(VisionTransformerSmall32, "imagenet")
add_model_to_registry(VisionTransformerBase16, "imagenet")
add_model_to_registry(VisionTransformerBase32, "imagenet")
add_model_to_registry(VisionTransformerLarge16, "imagenet")
add_model_to_registry(VisionTransformerLarge32, "imagenet")

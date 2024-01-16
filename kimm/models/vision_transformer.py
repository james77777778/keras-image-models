import typing

import keras
from keras import backend
from keras import layers
from keras import utils
from keras.src.applications import imagenet_utils

from kimm import layers as kimm_layers
from kimm.blocks import apply_transformer_block
from kimm.models.base_model import BaseModel
from kimm.utils import add_model_to_registry


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
        **kwargs,
    ):
        # Prepare feature extraction
        features = {}

        # Determine proper input shape
        input_shape = imagenet_utils.obtain_input_shape(
            input_shape,
            default_size=384,
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

        # [0, 255] to [-1, 1]
        if include_preprocessing:
            x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)(x)

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

        if include_top:
            x = x[:, 0]  # class token
            x = layers.Dropout(dropout_rate, name="head_drop")(x)
            x = layers.Dense(
                classes, activation=classifier_activation, name="head"
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

        super().__init__(inputs=inputs, outputs=x, features=features, **kwargs)

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.use_qkv_bias = use_qkv_bias
        self.use_qk_norm = use_qk_norm
        self.include_preprocessing = include_preprocessing
        self.include_top = include_top
        self.pooling = pooling
        self.dropout_rate = dropout_rate
        self.classes = classes
        self.classifier_activation = classifier_activation
        self._weights = weights  # `self.weights` is been used internally

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

    def fix_config(self, config):
        unused_kwargs = [
            "patch_size",
            "embed_dim",
            "depth",
            "num_heads",
            "mlp_ratio",
            "use_qkv_bias",
            "use_qk_norm",
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
            input_tensor,
            input_shape,
            include_preprocessing,
            include_top,
            pooling,
            pos_dropout_rate,
            dropout_rate,
            classes,
            classifier_activation,
            weights,
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
            input_tensor,
            input_shape,
            include_preprocessing,
            include_top,
            pooling,
            pos_dropout_rate,
            dropout_rate,
            classes,
            classifier_activation,
            weights,
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
            input_tensor,
            input_shape,
            include_preprocessing,
            include_top,
            pooling,
            pos_dropout_rate,
            dropout_rate,
            classes,
            classifier_activation,
            weights,
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
            input_tensor,
            input_shape,
            include_preprocessing,
            include_top,
            pooling,
            pos_dropout_rate,
            dropout_rate,
            classes,
            classifier_activation,
            weights,
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
            input_tensor,
            input_shape,
            include_preprocessing,
            include_top,
            pooling,
            pos_dropout_rate,
            dropout_rate,
            classes,
            classifier_activation,
            weights,
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
            input_tensor,
            input_shape,
            include_preprocessing,
            include_top,
            pooling,
            pos_dropout_rate,
            dropout_rate,
            classes,
            classifier_activation,
            weights,
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
            input_tensor,
            input_shape,
            include_preprocessing,
            include_top,
            pooling,
            pos_dropout_rate,
            dropout_rate,
            classes,
            classifier_activation,
            weights,
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
            input_tensor,
            input_shape,
            include_preprocessing,
            include_top,
            pooling,
            pos_dropout_rate,
            dropout_rate,
            classes,
            classifier_activation,
            weights,
            name=name,
            **kwargs,
        )

    @staticmethod
    def available_feature_keys():
        feature_keys = ["EMBEDDING"]
        feature_keys.extend([f"BLOCK{i}" for i in range(24)])
        return feature_keys


add_model_to_registry(VisionTransformerTiny16, True)
add_model_to_registry(VisionTransformerTiny32, True)
add_model_to_registry(VisionTransformerSmall16, True)
add_model_to_registry(VisionTransformerSmall32, True)
add_model_to_registry(VisionTransformerBase16, True)
add_model_to_registry(VisionTransformerBase32, True)
add_model_to_registry(VisionTransformerLarge16, True)
add_model_to_registry(VisionTransformerLarge32, True)

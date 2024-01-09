import functools
import typing

import keras
from keras import backend
from keras import layers
from keras import utils
from keras.src.applications import imagenet_utils

from kimm import layers as kimm_layers
from kimm.models.feature_extractor import FeatureExtractor


def apply_mlp_block(
    inputs,
    hidden_dim,
    output_dim=None,
    activation="gelu",
    normalization=None,
    use_bias=True,
    dropout_rate=0.0,
    use_convolution=False,
    name="mlp_block",
):
    input_dim = inputs.shape[-1]
    output_dim = output_dim or input_dim

    x = inputs
    if use_convolution:
        x = layers.Conv2D(
            hidden_dim, 1, 1, use_bias=use_bias, name=f"{name}_fc1"
        )(x)
    else:
        x = layers.Dense(hidden_dim, use_bias=use_bias, name=f"{name}_fc1")(x)
    x = layers.Activation(activation, name=f"{name}_act")(x)
    x = layers.Dropout(dropout_rate, name=f"{name}_drop1")(x)
    if normalization is not None:
        x = normalization(name=f"{name}_norm")(x)
    if use_convolution:
        x = layers.Conv2D(
            output_dim, 1, 1, use_bias=use_bias, name=f"{name}_fc2"
        )(x)
    else:
        x = layers.Dense(output_dim, use_bias=use_bias, name=f"{name}_fc2")(x)
    x = layers.Dropout(dropout_rate, name=f"{name}_drop2")(x)
    return x


def apply_vision_transformer_block(
    inputs,
    dim,
    num_heads,
    mlp_ratio=4.0,
    use_qkv_bias=False,
    use_qk_norm=False,
    projection_dropout_rate=0.0,
    attention_dropout_rate=0.0,
    scale_initializer=None,
    activation="gelu",
    normalization=layers.LayerNormalization,
    name="vision_transformer_block",
):
    x = inputs
    residual1 = x

    x = normalization(name=f"{name}_norm1")(x)
    x = kimm_layers.Attention(
        dim,
        num_heads,
        use_qkv_bias,
        use_qk_norm,
        attention_dropout_rate,
        projection_dropout_rate,
        name=f"{name}_attn",
    )(x)
    if scale_initializer is not None:
        x = kimm_layers.LayerScale(
            dim, initializer=scale_initializer, name=f"{name}_ls1"
        )(x)
    # TODO: add DropPath
    x = layers.Add()([residual1, x])

    residual2 = x
    x = normalization(name=f"{name}_norm2")(x)
    x = apply_mlp_block(
        x,
        int(dim * mlp_ratio),
        activation=activation,
        dropout_rate=projection_dropout_rate,
        name=f"{name}_mlp",
    )
    if scale_initializer is not None:
        x = kimm_layers.LayerScale(
            dim, initializer=scale_initializer, name=f"{name}_ls2"
        )(x)
    # TODO: add DropPath
    x = layers.Add()([residual2, x])
    return x


class VisionTransformer(FeatureExtractor):
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
        features["Depth0"] = x
        x = layers.Dropout(pos_dropout_rate, name="pos_dropout")(x)

        for i in range(depth):
            x = apply_vision_transformer_block(
                x,
                embed_dim,
                num_heads,
                mlp_ratio,
                use_qkv_bias,
                use_qk_norm,
                activation="gelu",
                normalization=functools.partial(
                    layers.LayerNormalization, epsilon=1e-6
                ),
                name=f"blocks_{i}",
            )
            features[f"Depth{i + 1}"] = x
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
        raise NotImplementedError

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

import typing
import warnings

import keras
from keras import backend
from keras import layers
from keras import ops

from kimm._src.blocks.transformer import apply_transformer_block
from kimm._src.kimm_export import kimm_export
from kimm._src.layers.position_embedding import PositionEmbedding
from kimm._src.models.base_model import BaseModel
from kimm._src.utils.model_registry import add_model_to_registry


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
        kwargs["weights_url"] = self.get_weights_url(kwargs["weights"])

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

        # Patch embedding
        x = layers.Conv2D(
            embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            use_bias=True,
            name="patch_embed_conv",
        )(x)

        # TODO: natively support channels_first
        if backend.image_data_format() == "channels_first":
            x = ops.transpose(x, [0, 2, 3, 1])

        x = layers.Reshape((-1, embed_dim))(x)
        x = PositionEmbedding(name="postition_embedding")(x)
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


# Model Definition


class VisionTransformerVariant(VisionTransformer):
    # Parameters
    patch_size = None
    embed_dim = None
    depth = None
    num_heads = None
    mlp_ratio = None
    use_qkv_bias = None
    use_qk_norm = None
    pos_dropout_rate = None

    def __init__(
        self,
        input_tensor: keras.KerasTensor = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        include_preprocessing: bool = True,
        include_top: bool = True,
        pooling: typing.Optional[str] = None,
        dropout_rate: float = 0.1,
        classes: int = 1000,
        classifier_activation: str = "softmax",
        weights: typing.Optional[str] = "imagenet",
        name: typing.Optional[str] = None,
        **kwargs,
    ):
        if type(self) is VisionTransformerVariant:
            raise NotImplementedError(
                f"Cannot instantiate base class: {self.__class__.__name__}. "
                "You should use its subclasses."
            )
        kwargs = self.fix_config(kwargs)
        if len(getattr(self, "available_weights", [])) == 0:
            warnings.warn(
                f"{self.__class__.__name__} doesn't have pretrained weights "
                f"for '{weights}'."
            )
            weights = None
        super().__init__(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            use_qkv_bias=self.use_qkv_bias,
            use_qk_norm=self.use_qk_norm,
            pos_dropout_rate=self.pos_dropout_rate,
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


@kimm_export(parent_path=["kimm.models", "kimm.models.vision_transformer"])
class VisionTransformerTiny16(VisionTransformerVariant):
    available_feature_keys = [
        "EMBEDDING",
        *[f"BLOCK{i}" for i in range(12)],
    ]
    available_weights = [
        (
            "imagenet",
            VisionTransformer.default_origin,
            "visiontransformertiny16_vit_tiny_patch16_384.keras",
        )
    ]

    # Parameters
    patch_size = 16
    embed_dim = 192
    depth = 12
    num_heads = 3
    mlp_ratio = 4.0
    use_qkv_bias = True
    use_qk_norm = False
    pos_dropout_rate = 0.0


@kimm_export(parent_path=["kimm.models", "kimm.models.vision_transformer"])
class VisionTransformerTiny32(VisionTransformerVariant):
    available_feature_keys = [
        "EMBEDDING",
        *[f"BLOCK{i}" for i in range(12)],
    ]
    available_weights = []

    # Parameters
    patch_size = 32
    embed_dim = 192
    depth = 12
    num_heads = 3
    mlp_ratio = 4.0
    use_qkv_bias = True
    use_qk_norm = False
    pos_dropout_rate = 0.0


@kimm_export(parent_path=["kimm.models", "kimm.models.vision_transformer"])
class VisionTransformerSmall16(VisionTransformerVariant):
    available_feature_keys = [
        "EMBEDDING",
        *[f"BLOCK{i}" for i in range(12)],
    ]
    available_weights = [
        (
            "imagenet",
            VisionTransformer.default_origin,
            "visiontransformersmall16_vit_small_patch16_384.keras",
        )
    ]

    # Parameters
    patch_size = 16
    embed_dim = 384
    depth = 12
    num_heads = 6
    mlp_ratio = 4.0
    use_qkv_bias = True
    use_qk_norm = False
    pos_dropout_rate = 0.0


@kimm_export(parent_path=["kimm.models", "kimm.models.vision_transformer"])
class VisionTransformerSmall32(VisionTransformerVariant):
    available_feature_keys = [
        "EMBEDDING",
        *[f"BLOCK{i}" for i in range(12)],
    ]
    available_weights = [
        (
            "imagenet",
            VisionTransformer.default_origin,
            "visiontransformersmall32_vit_small_patch32_384.keras",
        )
    ]

    # Parameters
    patch_size = 32
    embed_dim = 384
    depth = 12
    num_heads = 6
    mlp_ratio = 4.0
    use_qkv_bias = True
    use_qk_norm = False
    pos_dropout_rate = 0.0


@kimm_export(parent_path=["kimm.models", "kimm.models.vision_transformer"])
class VisionTransformerBase16(VisionTransformerVariant):
    available_feature_keys = [
        "EMBEDDING",
        *[f"BLOCK{i}" for i in range(12)],
    ]
    available_weights = [
        (
            "imagenet",
            VisionTransformer.default_origin,
            "visiontransformerbase16_vit_base_patch16_384.keras",
        )
    ]

    # Parameters
    patch_size = 16
    embed_dim = 768
    depth = 12
    num_heads = 12
    mlp_ratio = 4.0
    use_qkv_bias = True
    use_qk_norm = False
    pos_dropout_rate = 0.0


@kimm_export(parent_path=["kimm.models", "kimm.models.vision_transformer"])
class VisionTransformerBase32(VisionTransformerVariant):
    available_feature_keys = [
        "EMBEDDING",
        *[f"BLOCK{i}" for i in range(12)],
    ]
    available_weights = [
        (
            "imagenet",
            VisionTransformer.default_origin,
            "visiontransformerbase32_vit_base_patch32_384.keras",
        )
    ]

    # Parameters
    patch_size = 32
    embed_dim = 768
    depth = 12
    num_heads = 12
    mlp_ratio = 4.0
    use_qkv_bias = True
    use_qk_norm = False
    pos_dropout_rate = 0.0


@kimm_export(parent_path=["kimm.models", "kimm.models.vision_transformer"])
class VisionTransformerLarge16(VisionTransformerVariant):
    available_feature_keys = [
        "EMBEDDING",
        *[f"BLOCK{i}" for i in range(24)],
    ]
    available_weights = []

    # Parameters
    patch_size = 16
    embed_dim = 1024
    depth = 24
    num_heads = 16
    mlp_ratio = 4.0
    use_qkv_bias = True
    use_qk_norm = False
    pos_dropout_rate = 0.0


@kimm_export(parent_path=["kimm.models", "kimm.models.vision_transformer"])
class VisionTransformerLarge32(VisionTransformerVariant):
    available_feature_keys = [
        "EMBEDDING",
        *[f"BLOCK{i}" for i in range(24)],
    ]
    available_weights = []

    # Parameters
    patch_size = 32
    embed_dim = 1024
    depth = 24
    num_heads = 16
    mlp_ratio = 4.0
    use_qkv_bias = True
    use_qk_norm = False
    pos_dropout_rate = 0.0


add_model_to_registry(VisionTransformerTiny16, "imagenet")
add_model_to_registry(VisionTransformerTiny32)
add_model_to_registry(VisionTransformerSmall16, "imagenet")
add_model_to_registry(VisionTransformerSmall32, "imagenet")
add_model_to_registry(VisionTransformerBase16, "imagenet")
add_model_to_registry(VisionTransformerBase32, "imagenet")
add_model_to_registry(VisionTransformerLarge16)
add_model_to_registry(VisionTransformerLarge32)

import keras
from keras import layers
from keras import ops


@keras.saving.register_keras_serializable(package="kimm")
class Attention(layers.Layer):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        use_qkv_bias: bool = False,
        use_qk_norm: bool = False,
        attention_dropout_rate: float = 0.0,
        projection_dropout_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.use_qkv_bias = use_qkv_bias
        self.use_qk_norm = use_qk_norm
        self.attention_dropout_rate = attention_dropout_rate
        self.projection_dropout_rate = projection_dropout_rate

        self.qkv = layers.Dense(
            hidden_dim * 3,
            use_bias=use_qkv_bias,
            dtype=self.dtype_policy,
            name=f"{self.name}_qkv",
        )
        if use_qk_norm:
            self.q_norm = layers.LayerNormalization(
                dtype=self.dtype_policy, name=f"{self.name}_q_norm"
            )
            self.k_norm = layers.LayerNormalization(
                dtype=self.dtype_policy, name=f"{self.name}_k_norm"
            )
        else:
            self.q_norm = layers.Identity(dtype=self.dtype_policy)
            self.k_norm = layers.Identity(dtype=self.dtype_policy)

        self.attention_dropout = layers.Dropout(
            attention_dropout_rate,
            dtype=self.dtype_policy,
            name=f"{self.name}_attn_drop",
        )
        self.projection = layers.Dense(
            hidden_dim, dtype=self.dtype_policy, name=f"{self.name}_proj"
        )
        self.projection_dropout = layers.Dropout(
            projection_dropout_rate,
            dtype=self.dtype_policy,
            name=f"{self.name}_proj_drop",
        )

    def build(self, input_shape):
        self.qkv.build(input_shape)
        qkv_output_shape = list(input_shape)
        qkv_output_shape[-1] = qkv_output_shape[-1] * 3
        self.q_norm.build(qkv_output_shape)
        self.k_norm.build(qkv_output_shape)
        attention_input_shape = [
            input_shape[0],
            self.num_heads,
            input_shape[1],
            input_shape[1],
        ]
        self.attention_dropout.build(attention_input_shape)
        self.projection.build(input_shape)
        self.projection_dropout.build(input_shape)
        self.built = True

    def call(self, inputs, training=None, mask=None):
        input_shape = ops.shape(inputs)
        qkv = self.qkv(inputs)
        qkv = ops.reshape(
            qkv,
            [
                input_shape[0],
                input_shape[1],
                3,
                self.num_heads,
                self.head_dim,
            ],
        )
        qkv = ops.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = ops.unstack(qkv, 3, axis=0)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # attention
        q = ops.multiply(q, self.scale)
        attn = ops.matmul(q, ops.swapaxes(k, -2, -1))
        attn = ops.softmax(attn)
        attn = self.attention_dropout(attn)
        x = ops.matmul(attn, v)

        x = ops.swapaxes(x, 1, 2)
        x = ops.reshape(x, input_shape)
        x = self.projection(x)
        x = self.projection_dropout(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "use_qkv_bias": self.use_qkv_bias,
                "use_qk_norm": self.use_qk_norm,
                "attention_dropout_rate": self.attention_dropout_rate,
                "projection_dropout_rate": self.projection_dropout_rate,
                "name": self.name,
            }
        )
        return config

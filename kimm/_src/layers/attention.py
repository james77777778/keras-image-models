import keras
from keras import InputSpec
from keras import layers
from keras import ops

from kimm._src.kimm_export import kimm_export


@kimm_export(parent_path=["kimm.layers"])
@keras.saving.register_keras_serializable(package="kimm")
class Attention(layers.Layer):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        use_qkv_bias: bool = False,
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
        self.attention_dropout_rate = attention_dropout_rate
        self.projection_dropout_rate = projection_dropout_rate

        self.qkv = layers.Dense(
            hidden_dim * 3,
            use_bias=use_qkv_bias,
            dtype=self.dtype_policy,
            name=f"{self.name}_qkv",
        )

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
        self.input_spec = InputSpec(ndim=len(input_shape))
        if self.input_spec.ndim not in (3, 4):
            raise ValueError(
                "The ndim of the inputs must be 3 or 4. "
                f"Received: input_shape={input_shape}"
            )

        self.qkv.build(input_shape)
        qkv_output_shape = list(input_shape)
        qkv_output_shape[-1] = qkv_output_shape[-1] * 3
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
        if self.input_spec.ndim == 3:
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
            qkv = ops.transpose(qkv, [0, 3, 2, 1, 4])
            q, k, v = ops.unstack(qkv, 3, axis=2)
        else:
            # self.input_spec.ndim==4
            qkv = ops.reshape(
                qkv,
                [
                    input_shape[0],
                    input_shape[1],
                    input_shape[2],
                    3,
                    self.num_heads,
                    self.head_dim,
                ],
            )
            qkv = ops.transpose(qkv, [0, 1, 4, 3, 2, 5])
            q, k, v = ops.unstack(qkv, 3, axis=3)

        # attention
        q = ops.multiply(q, self.scale)
        attn = ops.matmul(q, ops.swapaxes(k, -2, -1))
        attn = ops.softmax(attn)
        attn = self.attention_dropout(attn)
        x = ops.matmul(attn, v)
        x = ops.reshape(ops.swapaxes(x, -3, -2), input_shape)
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
                "attention_dropout_rate": self.attention_dropout_rate,
                "projection_dropout_rate": self.projection_dropout_rate,
                "name": self.name,
            }
        )
        return config

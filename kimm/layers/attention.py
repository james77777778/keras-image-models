from keras import layers
from keras import ops


class Attention(layers.Layer):
    def __init__(
        self,
        hidden_dim,
        num_heads=8,
        use_qkv_bias=False,
        use_qk_norm=False,
        attention_dropout_rate=0.0,
        projection_dropout_rate=0.0,
        name="attention",
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
        self.name = name

        self.qkv = layers.Dense(
            hidden_dim * 3, use_bias=use_qkv_bias, name=f"{name}_qkv"
        )
        if use_qk_norm:
            self.q_norm = layers.LayerNormalization(name=f"{name}_q_norm")
            self.k_norm = layers.LayerNormalization(name=f"{name}_k_norm")
        else:
            self.q_norm = layers.Identity()
            self.k_norm = layers.Identity()

        self.attention_dropout = layers.Dropout(
            attention_dropout_rate, name=f"{name}_attn_drop"
        )
        self.projection = layers.Dense(hidden_dim, name=f"{name}_proj")
        self.projection_dropout = layers.Dropout(
            projection_dropout_rate, name=f"{name}_proj_drop"
        )

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


if __name__ == "__main__":
    from keras import models
    from keras import random

    inputs = layers.Input(shape=[197, 768])
    outputs = Attention(768)(inputs)

    model = models.Model(inputs, outputs)
    model.summary()

    inputs = random.uniform([1, 197, 768])
    outputs = model(inputs)
    print(outputs.shape)

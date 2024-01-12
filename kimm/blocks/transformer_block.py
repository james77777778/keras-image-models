from keras import layers

from kimm import layers as kimm_layers


def apply_mlp_block(
    inputs,
    hidden_dim,
    output_dim=None,
    activation="gelu",
    normalization=None,
    use_bias=True,
    dropout_rate=0.0,
    name="mlp_block",
):
    input_dim = inputs.shape[-1]
    output_dim = output_dim or input_dim

    x = inputs
    x = layers.Dense(hidden_dim, use_bias=use_bias, name=f"{name}_fc1")(x)
    x = layers.Activation(activation, name=f"{name}_act")(x)
    x = layers.Dropout(dropout_rate, name=f"{name}_drop1")(x)
    if normalization is not None:
        x = normalization(name=f"{name}_norm")(x)
    x = layers.Dense(output_dim, use_bias=use_bias, name=f"{name}_fc2")(x)
    x = layers.Dropout(dropout_rate, name=f"{name}_drop2")(x)
    return x


def apply_transformer_block(
    inputs,
    dim,
    num_heads,
    mlp_ratio=4.0,
    use_qkv_bias=False,
    use_qk_norm=False,
    projection_dropout_rate=0.0,
    attention_dropout_rate=0.0,
    activation="gelu",
    name="transformer_block",
):
    x = inputs
    residual_1 = x

    x = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm1")(x)
    x = kimm_layers.Attention(
        dim,
        num_heads,
        use_qkv_bias,
        use_qk_norm,
        attention_dropout_rate,
        projection_dropout_rate,
        name=f"{name}_attn",
    )(x)
    x = layers.Add()([residual_1, x])

    residual_2 = x
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm2")(x)
    x = apply_mlp_block(
        x,
        int(dim * mlp_ratio),
        activation=activation,
        dropout_rate=projection_dropout_rate,
        name=f"{name}_mlp",
    )
    x = layers.Add()([residual_2, x])
    return x

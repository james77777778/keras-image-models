import keras
from keras import layers
from keras import ops


@keras.saving.register_keras_serializable(package="kimm")
class LearnableAffine(layers.Layer):
    def __init__(self, scale_value=1.0, bias_value=0.0, **kwargs):
        super().__init__(**kwargs)
        if isinstance(scale_value, int):
            raise ValueError(
                f"scale_value must be a integer. Received: {scale_value}"
            )
        if isinstance(bias_value, int):
            raise ValueError(
                f"bias_value must be a integer. Received: {bias_value}"
            )
        self.scale_value = scale_value
        self.bias_value = bias_value

    def build(self, input_shape):
        self.scale = self.add_weight(
            shape=(1,),
            initializer=lambda shape, dtype: ops.cast(self.scale_value, dtype),
            trainable=True,
            name="scale",
        )
        self.bias = self.add_weight(
            shape=(1,),
            initializer=lambda shape, dtype: ops.cast(self.bias_value, dtype),
            trainable=True,
            name="bias",
        )
        self.built = True

    def call(self, inputs, training=None, mask=None):
        return ops.add(ops.multiply(inputs, self.scale), self.bias)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "scale_value": self.scale_value,
                "bias_value": self.bias_value,
                "name": self.name,
            }
        )
        return config

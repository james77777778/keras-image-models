from keras import initializers
from keras import layers
from keras import ops


class LayerScale(layers.Layer):
    def __init__(
        self,
        hidden_size,
        initializer=initializers.Constant(1e-5),
        name="layer_scale",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.initializer = initializer
        self.name = name

    def build(self, input_shape):
        self.gamma = self.add_weight(
            [self.hidden_size], initializer=self.initializer, name="gamma"
        )
        self.built = True

    def call(self, inputs, training=None, mask=None):
        return ops.multiply(inputs, self.gamma)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "initializer": initializers.serialize(self.initializer),
                "name": self.name,
            }
        )
        return config

import typing

import keras
from keras import backend
from keras import layers
from keras import ops


@keras.saving.register_keras_serializable(package="kimm")
class GlobalResponseNormalization(layers.Layer):
    def __init__(
        self,
        channel_axis: typing.Union[int, typing.Sequence[int]] = -1,
        spatial_axis: typing.Union[int, typing.Sequence[int]] = (1, 2),
        epsilon: float = backend.epsilon(),
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(channel_axis, int):
            channel_axis = (channel_axis,)
        if isinstance(spatial_axis, int):
            spatial_axis = (spatial_axis,)
        self.channel_axis = channel_axis
        self.spatial_axis = spatial_axis
        self.epsilon = epsilon

    def build(self, input_shape):
        channel_axis = [
            a if a >= 0 else a + len(input_shape) for a in self.channel_axis
        ]
        shape = []
        for i, dim in enumerate(input_shape):
            if i in channel_axis:
                shape.append(dim)
        self.gamma = self.add_weight(
            shape=shape, name="gamma", initializer="zeros", trainable=True
        )
        self.beta = self.add_weight(
            shape=shape, name="beta", initializer="zeros", trainable=True
        )
        self.built = True

    def call(self, inputs, training=None, mask=None):
        x = inputs
        # gx = torch.norm(X, p=2, dim=(1,2), keepdim=True)
        x_g = ops.norm(x, ord=2, axis=self.spatial_axis, keepdims=True)
        # nx = gx / (gx.mean(dim=-1, keepdim=True)+1e-6)
        x_n = ops.divide(
            x_g,
            ops.add(
                ops.mean(x_g, axis=self.channel_axis, keepdims=True),
                self.epsilon,
            ),
        )
        # gamma * (X * nx) + beta + X
        x = ops.add(
            ops.add(ops.multiply(ops.multiply(x, x_n), self.gamma), self.beta),
            x,
        )
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "channel_axis": self.channel_axis,
                "spatial_axis": self.spatial_axis,
                "epsilon": self.epsilon,
                "name": self.name,
            }
        )
        return config

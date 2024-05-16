import keras
from keras import initializers
from keras import layers
from keras import ops

from kimm._src.kimm_export import kimm_export


@kimm_export(parent_path=["kimm.layers"])
@keras.saving.register_keras_serializable(package="kimm")
class LayerScale(layers.Layer):
    def __init__(
        self,
        axis: int = -1,
        initializer: initializers.Initializer = initializers.Constant(1e-5),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.axis = axis
        self.initializer = initializer

    def build(self, input_shape):
        if isinstance(self.axis, list):
            shape = tuple([input_shape[dim] for dim in self.axis])
        else:
            shape = (input_shape[self.axis],)
            self.axis = [self.axis]
        self.gamma = self.add_weight(
            shape, initializer=self.initializer, name="gamma"
        )
        self.built = True

    def call(self, inputs, training=None, mask=None):
        # Broadcasting only necessary for norm when the axis is not just
        # the last dimension
        input_shape = inputs.shape
        ndims = len(inputs.shape)
        broadcast_shape = [1] * ndims
        for dim in self.axis:
            broadcast_shape[dim] = input_shape[dim]
        gamma = ops.reshape(self.gamma, broadcast_shape)
        gamma = ops.cast(gamma, self.compute_dtype)
        return ops.multiply(inputs, gamma)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "axis": self.axis,
                "initializer": initializers.serialize(self.initializer),
                "name": self.name,
            }
        )
        return config

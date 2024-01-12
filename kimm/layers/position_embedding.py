from keras import layers
from keras import ops


class PositionEmbedding(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                "PositionEmbedding only accepts 3-dimensional input. "
                f"Received: input_shape={input_shape}"
            )
        self.pos_embed = self.add_weight(
            shape=[1, input_shape[-2] + 1, input_shape[-1]],
            initializer="random_normal",
            name="pos_embed",
        )
        self.cls_token = self.add_weight(
            shape=[1, 1, input_shape[-1]], initializer="zeros", name="cls_token"
        )
        self.built = True

    def call(self, inputs, training=None, mask=None):
        input_shape = ops.shape(inputs)
        x = ops.concatenate(
            [ops.tile(self.cls_token, [input_shape[0], 1, 1]), inputs],
            axis=1,
        )
        x = ops.add(x, self.pos_embed)
        return x

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[1] = output_shape[1] + 1
        return output_shape

    def get_config(self):
        return super().get_config()


if __name__ == "__main__":
    from keras import models
    from keras import random

    inputs = layers.Input([224, 224, 3])
    x = layers.Conv2D(
        768,
        16,
        16,
        use_bias=True,
    )(inputs)
    x = layers.Reshape((-1, 768))(x)
    outputs = PositionEmbedding()(x)

    model = models.Model(inputs, outputs)
    model.summary()

    inputs = random.uniform([1, 224, 224, 3])
    outputs = model(inputs)
    print(outputs.shape)

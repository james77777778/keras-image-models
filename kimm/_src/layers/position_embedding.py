import keras
from keras import layers
from keras import ops

from kimm._src.kimm_export import kimm_export


@kimm_export(parent_path=["kimm.layers"])
@keras.saving.register_keras_serializable(package="kimm")
class PositionEmbedding(layers.Layer):
    def __init__(self, height, width, **kwargs):
        super().__init__(**kwargs)
        # We need height and width for saving and loading
        self.height = int(height)
        self.width = int(width)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                "PositionEmbedding only accepts 3-dimensional input. "
                f"Received: input_shape={input_shape}"
            )
        if self.height * self.width != input_shape[-2]:
            raise ValueError(
                "The embedding size doesn't match the height and width. "
                f"Received: height={self.height}, width={self.width}, "
                f"input_shape={input_shape}"
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

    def save_own_variables(self, store):
        super().save_own_variables(store)
        # Add height and width information
        store["height"] = self.height
        store["width"] = self.width

    def load_own_variables(self, store):
        old_height = int(store["height"][...])
        old_width = int(store["width"][...])
        if old_height == self.height and old_width == self.width:
            self.pos_embed.assign(store["0"])
            self.cls_token.assign(store["1"])
            return

        # Resize the embedding if there is a shape mismatch
        pos_embed = store["0"]
        pos_embed_prefix, pos_embed = pos_embed[:, :1], pos_embed[:, 1:]
        pos_embed_dim = pos_embed.shape[-1]
        pos_embed = ops.cast(pos_embed, "float32")
        pos_embed = ops.reshape(pos_embed, [1, old_height, old_width, -1])
        pos_embed = ops.image.resize(
            pos_embed,
            size=[self.height, self.width],
            interpolation="bilinear",
            antialias=True,
            data_format="channels_last",
        )
        pos_embed = ops.reshape(pos_embed, [1, -1, pos_embed_dim])
        pos_embed = ops.concatenate([pos_embed_prefix, pos_embed], axis=1)
        self.pos_embed.assign(pos_embed)
        self.cls_token.assign(store["1"])

    def get_config(self):
        config = super().get_config()
        config.update(
            {"height": self.height, "width": self.width, "name": self.name}
        )
        return config

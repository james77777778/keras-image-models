import keras
import numpy as np
from keras import Sequential
from keras import layers
from keras import ops
from keras.src.backend import standardize_data_format
from keras.src.layers import Layer
from keras.src.utils.argument_validation import standardize_tuple


@keras.saving.register_keras_serializable(package="kimm")
class RepConv2D(Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding=None,
        has_skip: bool = True,
        reparameterized: bool = False,
        data_format=None,
        activation=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = standardize_tuple(kernel_size, 2, "kernel_size")
        self.strides = standardize_tuple(strides, 2, "strides")
        self.padding = padding
        self.has_skip = has_skip
        self._reparameterized = reparameterized
        self.data_format = standardize_data_format(data_format)
        self.activation = activation

        if self.kernel_size[0] != self.kernel_size[1]:
            raise ValueError(
                "The value of kernel_size must be the same. "
                f"Received: kernel_size={kernel_size}"
            )
        if self.strides[0] != self.strides[1]:
            raise ValueError(
                "The value of strides must be the same. "
                f"Received: strides={strides}"
            )
        if has_skip is True and (self.strides[0] != 1 or self.strides[1] != 1):
            raise ValueError(
                "strides must be `1` when `has_skip=True`. "
                f"Received: has_skip={has_skip}, strides={strides}"
            )

        self.zero_padding = layers.Identity(dtype=self.dtype_policy)
        if padding is None:
            padding = "same"
            if self.strides[0] > 1:
                padding = "valid"
                self.zero_padding = layers.ZeroPadding2D(
                    (self.kernel_size[0] // 2, self.kernel_size[1] // 2),
                    data_format=self.data_format,
                    dtype=self.dtype_policy,
                    name=f"{self.name}_pad",
                )
            self.padding = padding
        else:
            self.padding = padding

        channel_axis = -1 if self.data_format == "channels_last" else -3
        if self._reparameterized:
            self.rep_conv2d = layers.Conv2D(
                self.filters,
                self.kernel_size,
                self.strides,
                self.padding,
                data_format=self.data_format,
                use_bias=True,
                dtype=self.dtype_policy,
                name=f"{self.name}_reparam_conv",
            )
            self.identity = None
            self.conv_kxk = None
            self.conv_1x1 = None
        else:
            self.rep_conv2d = None
            if self.has_skip:
                self.identity = layers.BatchNormalization(
                    axis=channel_axis,
                    momentum=0.9,
                    epsilon=1e-5,
                    dtype=self.dtype_policy,
                    name=f"{self.name}_identity",
                )
            else:
                self.identity = None
            self.conv_kxk = Sequential(
                [
                    layers.Conv2D(
                        self.filters,
                        self.kernel_size,
                        self.strides,
                        padding=self.padding,
                        data_format=self.data_format,
                        use_bias=False,
                        dtype=self.dtype_policy,
                    ),
                    layers.BatchNormalization(
                        axis=channel_axis,
                        momentum=0.9,
                        epsilon=1e-5,
                        dtype=self.dtype_policy,
                    ),
                ],
                name=f"{self.name}_conv_kxk",
            )
            self.conv_1x1 = Sequential(
                [
                    layers.Conv2D(
                        self.filters,
                        1,
                        self.strides,
                        padding=self.padding,
                        data_format=self.data_format,
                        use_bias=False,
                        dtype=self.dtype_policy,
                    ),
                    layers.BatchNormalization(
                        axis=channel_axis,
                        momentum=0.9,
                        epsilon=1e-5,
                        dtype=self.dtype_policy,
                    ),
                ],
                name=f"{self.name}_conv_1x1",
            )

        if activation is None:
            self.act = layers.Identity(dtype=self.dtype_policy)
        else:
            self.act = layers.Activation(activation, dtype=self.dtype_policy)

        # Internal parameters for `_get_reparameterized_weights_from_layer`
        self._input_channels = None
        self._rep_kernel_shape = None

        # Attach extra layers
        self.extra_layers = []
        if self.rep_conv2d is not None:
            self.extra_layers.append(self.rep_conv2d)
        if self.identity is not None:
            self.extra_layers.append(self.identity)
        if self.conv_kxk is not None:
            self.extra_layers.append(self.conv_kxk)
        if self.conv_1x1 is not None:
            self.extra_layers.append(self.conv_1x1)
        self.extra_layers.append(self.act)

    def build(self, input_shape):
        channel_axis = -1 if self.data_format == "channels_last" else -3

        if isinstance(self.zero_padding, layers.ZeroPadding2D):
            padded_shape = self.zero_padding.compute_output_shape(input_shape)
        else:
            padded_shape = input_shape

        if self.rep_conv2d is not None:
            self.rep_conv2d.build(padded_shape)
        if self.identity is not None:
            self.identity.build(input_shape)
        if self.conv_kxk is not None:
            self.conv_kxk.build(padded_shape)
        if self.conv_1x1 is not None:
            self.conv_1x1.build(input_shape)

        # Update internal parameters
        self._input_channels = input_shape[channel_axis]
        if self.conv_kxk is not None:
            self._rep_kernel_shape = self.conv_kxk.layers[0].kernel.shape

        self.built = True

    def call(self, inputs, **kwargs):
        x = ops.cast(inputs, self.compute_dtype)
        padded_x = self.zero_padding(x)

        # Shortcut for reparameterized mode
        if self._reparameterized:
            return self.act(self.rep_conv2d(padded_x, **kwargs))

        if self.identity is None:
            x = self.conv_1x1(x, **kwargs) + self.conv_kxk(padded_x, **kwargs)
        else:
            identity = self.identity(x, **kwargs)
            x = self.conv_1x1(x, **kwargs) + self.conv_kxk(padded_x, **kwargs)
            x = x + identity
        return self.act(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "has_skip": self.has_skip,
                "reparameterized": self._reparameterized,
                "data_format": self.data_format,
                "activation": self.activation,
                "name": self.name,
            }
        )
        return config

    def _get_reparameterized_weights_from_layer(self, layer):
        if isinstance(layer, Sequential):
            if not isinstance(layer.layers[0], layers.Conv2D):
                raise ValueError
            if not isinstance(layer.layers[1], layers.BatchNormalization):
                raise ValueError
            kernel = ops.convert_to_numpy(layer.layers[0].kernel)
            gamma = ops.convert_to_numpy(layer.layers[1].gamma)
            beta = ops.convert_to_numpy(layer.layers[1].beta)
            running_mean = ops.convert_to_numpy(layer.layers[1].moving_mean)
            running_var = ops.convert_to_numpy(layer.layers[1].moving_variance)
            eps = layer.layers[1].epsilon
        elif isinstance(layer, layers.BatchNormalization):
            if self._rep_kernel_shape is None:
                raise ValueError(
                    "Remember to build the layer before performing"
                    "reparameterization. Failed to get valid "
                    "`self._rep_kernel_shape`."
                )
            # Calculate identity tensor
            kernel_value = ops.convert_to_numpy(
                ops.zeros(self._rep_kernel_shape)
            )
            kernel_value = kernel_value.copy()
            for i in range(self._input_channels):
                kernel_value[
                    self.kernel_size[0] // 2, self.kernel_size[1] // 2, i, i
                ] = 1
            kernel = kernel_value
            gamma = ops.convert_to_numpy(layer.gamma)
            beta = ops.convert_to_numpy(layer.beta)
            running_mean = ops.convert_to_numpy(layer.moving_mean)
            running_var = ops.convert_to_numpy(layer.moving_variance)
            eps = layer.epsilon

        # Use float64 for better precision
        kernel = kernel.astype("float64")
        gamma = gamma.astype("float64")
        beta = beta.astype("float64")
        running_var = running_var.astype("float64")
        running_var = running_var.astype("float64")

        std = np.sqrt(running_var + eps)
        t = np.reshape(gamma / std, [1, 1, 1, -1])
        return kernel * t, beta - running_mean * gamma / std

    def get_reparameterized_weights(self):
        kernel_1x1 = 0.0
        bias_1x1 = 0.0
        if self.conv_1x1 is not None:
            kernel_1x1, bias_1x1 = self._get_reparameterized_weights_from_layer(
                self.conv_1x1
            )
            pad = self.kernel_size[0] // 2
            kernel_1x1 = np.pad(
                kernel_1x1, [[pad, pad], [pad, pad], [0, 0], [0, 0]]
            )

        kernel_identity = 0.0
        bias_identity = 0.0
        if self.identity is not None:
            (
                kernel_identity,
                bias_identity,
            ) = self._get_reparameterized_weights_from_layer(self.identity)

        kernel_conv, bias_conv = self._get_reparameterized_weights_from_layer(
            self.conv_kxk
        )

        kernel_final = kernel_conv + kernel_1x1 + kernel_identity
        bias_final = bias_conv + bias_1x1 + bias_identity
        return kernel_final, bias_final

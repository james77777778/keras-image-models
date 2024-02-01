import typing

import keras
import numpy as np
from keras import Sequential
from keras import layers
from keras import ops
from keras.src.backend import standardize_data_format
from keras.src.layers import Layer
from keras.src.utils.argument_validation import standardize_tuple


@keras.saving.register_keras_serializable(package="kimm")
class MobileOneConv2D(Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding=None,
        has_skip: bool = True,
        use_depthwise: bool = False,
        branch_size: int = 1,
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
        self.use_depthwise = use_depthwise
        self.branch_size = branch_size
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

        # Build layers (rep_conv2d, identity, conv_kxk, conv_scale)
        self.rep_conv2d: typing.Optional[layers.Conv2D] = None
        self.identity: typing.Optional[layers.BatchNormalization] = None
        self.conv_kxk: typing.Optional[typing.List[Sequential]] = None
        self.conv_scale: typing.Optional[Sequential] = None
        if self._reparameterized:
            self.rep_conv2d = self._get_conv2d(
                use_depthwise,
                self.filters,
                self.kernel_size,
                self.strides,
                self.padding,
                use_bias=True,
                name=f"{self.name}_reparam_conv",
            )
        else:
            # Skip connection
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

            # Convoluation branches
            self.conv_kxk = []
            for i in range(self.branch_size):
                self.conv_kxk.append(
                    Sequential(
                        [
                            self._get_conv2d(
                                self.use_depthwise,
                                self.filters,
                                self.kernel_size,
                                self.strides,
                                self.padding,
                                use_bias=False,
                            ),
                            layers.BatchNormalization(
                                axis=channel_axis,
                                momentum=0.9,
                                epsilon=1e-5,
                                dtype=self.dtype_policy,
                            ),
                        ],
                        name=f"{self.name}_conv_kxk_{i}",
                    )
                )

            # Scale branch
            self.conv_scale = None
            if self.kernel_size[0] > 1:
                self.conv_scale = Sequential(
                    [
                        self._get_conv2d(
                            self.use_depthwise,
                            self.filters,
                            1,
                            self.strides,
                            self.padding,
                            use_bias=False,
                        ),
                        layers.BatchNormalization(
                            axis=channel_axis,
                            momentum=0.9,
                            epsilon=1e-5,
                            dtype=self.dtype_policy,
                        ),
                    ],
                    name=f"{self.name}_conv_scale",
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
            self.extra_layers.extend(self.conv_kxk)
        if self.conv_scale is not None:
            self.extra_layers.append(self.conv_scale)
        self.extra_layers.append(self.act)

    def _get_conv2d(
        self,
        use_depthwise,
        filters,
        kernel_size,
        strides,
        padding,
        use_bias,
        name=None,
    ):
        if use_depthwise:
            return layers.DepthwiseConv2D(
                kernel_size,
                strides,
                padding,
                data_format=self.data_format,
                use_bias=use_bias,
                dtype=self.dtype_policy,
                name=name,
            )
        else:
            return layers.Conv2D(
                filters,
                kernel_size,
                strides,
                padding,
                data_format=self.data_format,
                use_bias=use_bias,
                dtype=self.dtype_policy,
                name=name,
            )

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
            for layer in self.conv_kxk:
                layer.build(padded_shape)
        if self.conv_scale is not None:
            self.conv_scale.build(input_shape)

        # Update internal parameters
        self._input_channels = input_shape[channel_axis]
        if self.conv_kxk is not None:
            self._rep_kernel_shape = self.conv_kxk[0].layers[0].kernel.shape

        self.built = True

    def call(self, inputs, **kwargs):
        x = ops.cast(inputs, self.compute_dtype)
        padded_x = self.zero_padding(x)

        # Shortcut for reparameterized mode
        if self._reparameterized:
            return self.act(self.rep_conv2d(padded_x, **kwargs))

        # Skip connection
        identity_outputs = None
        if self.identity is not None:
            identity_outputs = self.identity(x, **kwargs)

        # Scale branch
        scale_outputs = None
        if self.conv_scale is not None:
            scale_outputs = self.conv_scale(x, **kwargs)

        # Conv branch
        conv_outputs = scale_outputs
        for layer in self.conv_kxk:
            if conv_outputs is None:
                conv_outputs = layer(padded_x, **kwargs)
            else:
                conv_outputs = layers.Add()(
                    [conv_outputs, layer(padded_x, **kwargs)]
                )

        if identity_outputs is not None:
            outputs = layers.Add()([conv_outputs, identity_outputs])
        else:
            outputs = conv_outputs
        return self.act(outputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "has_skip": self.has_skip,
                "use_depthwise": self.use_depthwise,
                "branch_size": self.branch_size,
                "reparameterized": self._reparameterized,
                "data_format": self.data_format,
                "activation": self.activation,
                "name": self.name,
            }
        )
        return config

    def _get_reparameterized_weights_from_layer(self, layer):
        if isinstance(layer, Sequential):
            if not isinstance(
                layer.layers[0], (layers.Conv2D, layers.DepthwiseConv2D)
            ):
                raise ValueError
            if not isinstance(layer.layers[1], layers.BatchNormalization):
                raise ValueError
            kernel = ops.convert_to_numpy(layer.layers[0].kernel)
            if self.use_depthwise:
                kernel = np.swapaxes(kernel, -2, -1)
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
            kernel = kernel_value.copy()
            if self.use_depthwise:
                kernel = np.swapaxes(kernel, -2, -1)
            for i in range(self._input_channels):
                group_i = 0 if self.use_depthwise else i
                kernel[
                    self.kernel_size[0] // 2,
                    self.kernel_size[1] // 2,
                    group_i,
                    i,
                ] = 1
            gamma = ops.convert_to_numpy(layer.gamma)
            beta = ops.convert_to_numpy(layer.beta)
            running_mean = ops.convert_to_numpy(layer.moving_mean)
            running_var = ops.convert_to_numpy(layer.moving_variance)
            eps = layer.epsilon

        # use float64 for better precision
        kernel = kernel.astype("float64")
        gamma = gamma.astype("float64")
        beta = beta.astype("float64")
        running_var = running_var.astype("float64")
        running_var = running_var.astype("float64")

        std = np.sqrt(running_var + eps)
        t = np.reshape(gamma / std, [1, 1, 1, -1])

        kernel_final = kernel * t
        if self.use_depthwise:
            kernel_final = np.swapaxes(kernel_final, -2, -1)
        return kernel_final, beta - running_mean * gamma / std

    def get_reparameterized_weights(self):
        # Get kernels and bias from scale branch
        kernel_scale = 0.0
        bias_scale = 0.0
        if self.conv_scale is not None:
            (
                kernel_scale,
                bias_scale,
            ) = self._get_reparameterized_weights_from_layer(self.conv_scale)
            pad = self.kernel_size[0] // 2
            kernel_scale = np.pad(
                kernel_scale, [[pad, pad], [pad, pad], [0, 0], [0, 0]]
            )

        # Get kernels and bias from skip branch
        kernel_identity = 0.0
        bias_identity = 0.0
        if self.identity is not None:
            (
                kernel_identity,
                bias_identity,
            ) = self._get_reparameterized_weights_from_layer(self.identity)

        # Get kernels and bias from conv branch
        kernel_conv = 0.0
        bias_conv = 0.0
        for i in range(self.branch_size):
            (
                _kernel_conv,
                _bias_conv,
            ) = self._get_reparameterized_weights_from_layer(self.conv_kxk[i])
            kernel_conv += _kernel_conv
            bias_conv += _bias_conv

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

import typing

import keras
import numpy as np
from keras import Sequential
from keras import layers
from keras import ops
from keras.src.backend import standardize_data_format
from keras.src.layers import Layer
from keras.src.utils.argument_validation import standardize_tuple

from kimm._src.kimm_export import kimm_export


@kimm_export(parent_path=["kimm.layers"])
@keras.saving.register_keras_serializable(package="kimm")
class ReparameterizableConv2D(Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding=None,
        has_skip: bool = True,
        has_scale: bool = True,
        use_depthwise: bool = False,
        branch_size: int = 1,
        reparameterized: bool = False,
        data_format=None,
        activation=None,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = standardize_tuple(kernel_size, 2, "kernel_size")
        self.strides = standardize_tuple(strides, 2, "strides")
        self.padding = padding
        self.has_skip = has_skip
        self.has_scale = has_scale
        self.use_depthwise = use_depthwise
        self.branch_size = branch_size
        self.reparameterized = reparameterized
        self.data_format = standardize_data_format(data_format)
        self.activation = activation

        if self.kernel_size[0] != self.kernel_size[1]:
            raise ValueError(
                "The values of `kernel_size` must be the same. "
                f"Received: kernel_size={kernel_size}"
            )
        if self.strides[0] != self.strides[1]:
            raise ValueError(
                "The values of `strides` must be the same. "
                f"Received: strides={strides}"
            )
        if has_skip is True and self.strides[0] != 1:
            raise ValueError(
                "When `has_skip=True`, `strides` must be `1`. "
                f"Received: has_skip={has_skip}, strides={strides}"
            )

        # Configure zero padding
        self.zero_padding: typing.Optional[layers.ZeroPadding2D] = None
        if self.padding is None:
            if self.strides[0] > 1 and self.kernel_size[0] > 1:
                self.padding = "valid"
                self.zero_padding = layers.ZeroPadding2D(
                    self.kernel_size[0] // 2,
                    data_format=self.data_format,
                    dtype=self.dtype_policy,
                    name=f"{self.name}_pad",
                )
            else:
                self.padding = "same"

        # Configure filters_axis
        self.filters_axis = -1 if self.data_format == "channels_last" else -3

        # Build layers
        bn_momentum, bn_epsilon = 0.9, 1e-5  # Defaults to torch's default

        self.reparameterized_conv2d: typing.Optional[layers.Conv2D] = None
        self.skip: typing.Optional[layers.BatchNormalization] = None
        self.conv_scale: typing.Optional[Sequential] = None
        self.conv_kxk: typing.List[Sequential] = []
        self.act: typing.Optional[layers.Activation] = None

        if self.reparameterized:
            self.reparameterized_conv2d = self._get_conv2d_layer(
                self.use_depthwise,
                self.filters,
                self.kernel_size,
                self.strides,
                self.padding,
                use_bias=True,
                name=f"{self.name}_reparam_conv",
            )
        else:
            # Skip branch
            if self.has_skip:
                self.skip = layers.BatchNormalization(
                    axis=self.filters_axis,
                    momentum=bn_momentum,
                    epsilon=bn_epsilon,
                    dtype=self.dtype_policy,
                    name=f"{self.name}_skip",
                )
            # Scale branch
            if self.has_scale:
                self.conv_scale = Sequential(
                    [
                        self._get_conv2d_layer(
                            self.use_depthwise,
                            self.filters,
                            1,
                            self.strides,
                            self.padding,
                            use_bias=False,
                        ),
                        layers.BatchNormalization(
                            axis=self.filters_axis,
                            momentum=bn_momentum,
                            epsilon=bn_epsilon,
                            dtype=self.dtype_policy,
                        ),
                    ],
                    name=f"{self.name}_conv_scale",
                )
            # Overparameterized branch
            for i in range(self.branch_size):
                self.conv_kxk.append(
                    Sequential(
                        [
                            self._get_conv2d_layer(
                                self.use_depthwise,
                                self.filters,
                                self.kernel_size,
                                self.strides,
                                self.padding,
                                use_bias=False,
                            ),
                            layers.BatchNormalization(
                                axis=self.filters_axis,
                                momentum=bn_momentum,
                                epsilon=bn_epsilon,
                                dtype=self.dtype_policy,
                            ),
                        ],
                        name=f"{self.name}_conv_kxk_{i}",
                    )
                )
        if activation is not None:
            self.act = layers.Activation(activation, dtype=self.dtype_policy)

    @property
    def _sublayers(self):
        """An internal api for weights exporting.

        Generally, you don't need this.
        """
        sublayers = []
        if self.reparameterized_conv2d is not None:
            sublayers.append(self.reparameterized_conv2d)
        if self.skip is not None:
            sublayers.append(self.skip)
        if self.conv_scale is not None:
            sublayers.append(self.conv_scale)
        if self.conv_kxk is not None:
            sublayers.extend(self.conv_kxk)
        if self.act is not None:
            sublayers.append(self.act)
        return sublayers

    def _get_conv2d_layer(
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
        input_filters = input_shape[self.filters_axis]
        if self.use_depthwise and input_filters != self.filters:
            raise ValueError(
                "When `use_depthwise=True`, `filters` must be the same as "
                f"input filters. Received: input_shape={input_shape}, "
                f"filters={self.filters}"
            )

        if isinstance(self.zero_padding, layers.ZeroPadding2D):
            input_shape = self.zero_padding.compute_output_shape(input_shape)

        if self.reparameterized_conv2d is not None:
            self.reparameterized_conv2d.build(input_shape)

        if self.skip is not None:
            self.skip.build(input_shape)
        if self.conv_scale is not None:
            self.conv_scale.build(input_shape)
        for layer in self.conv_kxk:
            layer.build(input_shape)

        # Update internal parameters
        self.input_filters = input_filters

        self.built = True

    def call(self, inputs, training=None, **kwargs):
        x = inputs
        padded_x = x

        if self.zero_padding is not None:
            padded_x = self.zero_padding(x)

        # Shortcut for reparameterized=True
        if self.reparameterized:
            y = self.reparameterized_conv2d(padded_x)
            if self.act is not None:
                y = self.act(y)
            return y

        # Skip branch
        y = None
        if self.skip is not None:
            y = self.skip(x, training=training)
        # Scale branch
        if self.conv_scale is not None:
            scale_y = self.conv_scale(x, training=training)
            if y is None:
                y = scale_y
            else:
                y = layers.Add(dtype=self.dtype_policy)([y, scale_y])
        # Overparameterized bracnh
        for idx in range(self.branch_size):
            over_y = self.conv_kxk[idx](padded_x, training=training)
            if y is None:
                y = over_y
            else:
                y = layers.Add(dtype=self.dtype_policy)([y, over_y])
        if self.act is not None:
            y = self.act(y)
        return y

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "has_skip": self.has_skip,
                "has_scale": self.has_scale,
                "use_depthwise": self.use_depthwise,
                "branch_size": self.branch_size,
                "reparameterized": self.reparameterized,
                "data_format": self.data_format,
                "activation": self.activation,
                "name": self.name,
            }
        )
        return config

    def _get_reparameterized_weights_from_layer(self, layer):
        if isinstance(layer, Sequential):
            # Check
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
            k = self.kernel_size[0]
            input_filters = 1 if self.use_depthwise else self.input_filters
            kernel = np.zeros(
                shape=[k, k, input_filters, self.filters], dtype="float64"
            )
            for i in range(self.input_filters):
                group_i = 0 if self.use_depthwise else i
                kernel[k // 2, k // 2, group_i, i] = 1
            gamma = ops.convert_to_numpy(layer.gamma)
            beta = ops.convert_to_numpy(layer.beta)
            running_mean = ops.convert_to_numpy(layer.moving_mean)
            running_var = ops.convert_to_numpy(layer.moving_variance)
            eps = layer.epsilon
        else:
            raise NotImplementedError

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
        # Get kernels and bias from skip branch
        kernel_identity = 0.0
        bias_identity = 0.0
        if self.skip is not None:
            (
                kernel_identity,
                bias_identity,
            ) = self._get_reparameterized_weights_from_layer(self.skip)

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

        # Get kernels and bias from overparameterized branch
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

import pytest
from absl.testing import parameterized
from keras import backend
from keras import random
from keras.src import testing

from kimm.layers.mobile_one_conv2d import MobileOneConv2D

TEST_CASES = [
    {
        "filters": 16,
        "kernel_size": 3,
        "has_skip": True,
        "use_depthwise": False,
        "branch_size": 2,
        "data_format": "channels_last",
        "input_shape": (1, 4, 4, 16),
        "output_shape": (1, 4, 4, 16),
        "num_trainable_weights": 11,
        "num_non_trainable_weights": 8,
    },
    {
        "filters": 16,
        "kernel_size": 3,
        "has_skip": True,
        "use_depthwise": True,
        "branch_size": 3,
        "data_format": "channels_last",
        "input_shape": (1, 4, 4, 16),
        "output_shape": (1, 4, 4, 16),
        "num_trainable_weights": 14,
        "num_non_trainable_weights": 10,
    },
    {
        "filters": 16,
        "kernel_size": 3,
        "has_skip": False,
        "use_depthwise": False,
        "branch_size": 2,
        "data_format": "channels_last",
        "input_shape": (1, 4, 4, 8),
        "output_shape": (1, 4, 4, 16),
        "num_trainable_weights": 9,
        "num_non_trainable_weights": 6,
    },
    {
        "filters": 16,
        "kernel_size": 5,
        "has_skip": True,
        "use_depthwise": False,
        "branch_size": 2,
        "data_format": "channels_last",
        "input_shape": (1, 4, 4, 16),
        "output_shape": (1, 4, 4, 16),
        "num_trainable_weights": 11,
        "num_non_trainable_weights": 8,
    },
    {
        "filters": 16,
        "kernel_size": 3,
        "has_skip": True,
        "use_depthwise": False,
        "branch_size": 2,
        "data_format": "channels_first",
        "input_shape": (1, 16, 4, 4),
        "output_shape": (1, 16, 4, 4),
        "num_trainable_weights": 11,
        "num_non_trainable_weights": 8,
    },
]


class MobileOneConv2DTest(testing.TestCase, parameterized.TestCase):
    @parameterized.parameters(TEST_CASES)
    @pytest.mark.requires_trainable_backend
    def test_mobile_one_conv2d_basic(
        self,
        filters,
        kernel_size,
        has_skip,
        use_depthwise,
        branch_size,
        data_format,
        input_shape,
        output_shape,
        num_trainable_weights,
        num_non_trainable_weights,
    ):
        if (
            backend.backend() == "tensorflow"
            and data_format == "channels_first"
        ):
            self.skipTest(
                "Conv2D in tensorflow backend with 'channels_first' is limited "
                "to be supported"
            )
        self.run_layer_test(
            MobileOneConv2D,
            init_kwargs={
                "filters": filters,
                "kernel_size": kernel_size,
                "has_skip": has_skip,
                "use_depthwise": use_depthwise,
                "branch_size": branch_size,
                "data_format": data_format,
            },
            input_shape=input_shape,
            expected_output_shape=output_shape,
            expected_num_trainable_weights=num_trainable_weights,
            expected_num_non_trainable_weights=num_non_trainable_weights,
            expected_num_losses=0,
            supports_masking=False,
        )

    @parameterized.parameters(TEST_CASES)
    def test_mobile_one_conv2d_get_reparameterized_weights(
        self,
        filters,
        kernel_size,
        has_skip,
        use_depthwise,
        branch_size,
        data_format,
        input_shape,
        output_shape,
        num_trainable_weights,
        num_non_trainable_weights,
    ):
        if (
            backend.backend() == "tensorflow"
            and data_format == "channels_first"
        ):
            self.skipTest(
                "Conv2D in tensorflow backend with 'channels_first' is limited "
                "to be supported"
            )
        layer = MobileOneConv2D(
            filters=filters,
            kernel_size=kernel_size,
            has_skip=has_skip,
            use_depthwise=use_depthwise,
            branch_size=branch_size,
            data_format=data_format,
        )
        layer.build(input_shape)
        reparameterized_layer = MobileOneConv2D(
            filters=filters,
            kernel_size=kernel_size,
            has_skip=has_skip,
            use_depthwise=use_depthwise,
            branch_size=branch_size,
            reparameterized=True,
            data_format=data_format,
        )
        reparameterized_layer.build(input_shape)
        x = random.uniform(input_shape)

        kernel, bias = layer.get_reparameterized_weights()
        reparameterized_layer.rep_conv2d.kernel.assign(kernel)
        reparameterized_layer.rep_conv2d.bias.assign(bias)
        y1 = layer(x, training=False)
        y2 = reparameterized_layer(x, training=False)

        self.assertAllClose(y1, y2, atol=1e-3)

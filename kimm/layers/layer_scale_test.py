import pytest
from absl.testing import parameterized
from keras.src import testing

from kimm.layers.layer_scale import LayerScale


class LayerScaleTest(testing.TestCase, parameterized.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_layer_scale_basic(self):
        self.run_layer_test(
            LayerScale,
            init_kwargs={"axis": -1},
            input_shape=(1, 10),
            expected_output_shape=(1, 10),
            expected_num_trainable_weights=1,
            expected_num_non_trainable_weights=0,
            expected_num_losses=0,
            supports_masking=False,
        )

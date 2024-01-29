from keras import random
from keras.src import testing

from kimm.models.regnet import RegNetX002
from kimm.models.repvgg import RepVGG
from kimm.utils.model_utils import get_reparameterized_model


class ModelUtilsTest(testing.TestCase):
    def test_get_reparameterized_model(self):
        # dummy RepVGG with random initialization
        model = RepVGG(
            [1, 1, 1, 1],
            [8, 8, 8, 8],
            8,
            include_preprocessing=False,
            weights=None,
        )
        reparameterized_model = get_reparameterized_model(model)
        x = random.uniform([1, 32, 32, 3])

        y1 = model(x, training=False)
        y2 = reparameterized_model(x, training=False)

        self.assertAllClose(y1, y2, atol=1e-5)

    def test_get_reparameterized_model_already(self):
        # dummy RepVGG with random initialization and reparameterized=True
        model = RepVGG(
            [1, 1, 1, 1],
            [8, 8, 8, 8],
            8,
            reparameterized=True,
            include_preprocessing=False,
            weights=None,
        )
        reparameterized_model = get_reparameterized_model(model)

        # same object
        self.assertEqual(id(model), id(reparameterized_model))

    def test_get_reparameterized_model_invalid(self):
        model = RegNetX002(weights=None)

        with self.assertRaisesRegex(
            ValueError, "There is no 'get_reparameterized_model' method"
        ):
            get_reparameterized_model(model)

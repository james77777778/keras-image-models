import keras
import pytest
import tensorflow as tf
from absl.testing import parameterized
from keras import backend
from keras import models
from keras import ops
from keras import random
from keras import utils
from keras.applications.imagenet_utils import decode_predictions
from keras.src import testing

from kimm import models as kimm_models
from kimm.utils import make_divisible

# name, class, default_size, features (name, shape),
# weights (defaults to imagenet)
MODEL_CONFIGS = [
    # convmixer
    (
        kimm_models.ConvMixer736D32.__name__,
        kimm_models.ConvMixer736D32,
        224,
        [
            ("STEM", [1, 32, 32, 768]),
            *((f"BLOCK{i}", [1, 32, 32, 768]) for i in range(32)),
        ],
    ),
    # convnext
    (
        kimm_models.ConvNeXtAtto.__name__,
        kimm_models.ConvNeXtAtto,
        288,
        [
            ("STEM_S4", [1, 72, 72, 40]),
            ("BLOCK0_S4", [1, 72, 72, 40]),
            ("BLOCK1_S8", [1, 36, 36, 80]),
            ("BLOCK2_S16", [1, 18, 18, 160]),
            ("BLOCK3_S32", [1, 9, 9, 320]),
        ],
    ),
    # densenet
    (
        kimm_models.DenseNet121.__name__,
        kimm_models.DenseNet121,
        224,
        [
            ("STEM_S4", [1, 56, 56, 64]),
            ("BLOCK0_S8", [1, 28, 28, 128]),
            ("BLOCK1_S16", [1, 14, 14, 256]),
            ("BLOCK2_S32", [1, 7, 7, 512]),
            ("BLOCK3_S32", [1, 7, 7, 1024]),
        ],
    ),
    # efficientnet
    (
        kimm_models.EfficientNetB2.__name__,
        kimm_models.EfficientNetB2,
        260,
        [
            ("STEM_S2", [1, 130, 130, make_divisible(32 * 1.1)]),
            ("BLOCK1_S4", [1, 65, 65, make_divisible(24 * 1.1)]),
            ("BLOCK2_S8", [1, 33, 33, make_divisible(40 * 1.1)]),
            ("BLOCK3_S16", [1, 17, 17, make_divisible(80 * 1.1)]),
            ("BLOCK5_S32", [1, 9, 9, make_divisible(192 * 1.1)]),
        ],
    ),
    (
        kimm_models.EfficientNetLiteB2.__name__,
        kimm_models.EfficientNetLiteB2,
        260,
        [
            ("STEM_S2", [1, 130, 130, make_divisible(32 * 1.1)]),
            ("BLOCK1_S4", [1, 65, 65, make_divisible(24 * 1.1)]),
            ("BLOCK2_S8", [1, 33, 33, make_divisible(40 * 1.1)]),
            ("BLOCK3_S16", [1, 17, 17, make_divisible(80 * 1.1)]),
            ("BLOCK5_S32", [1, 9, 9, make_divisible(192 * 1.1)]),
        ],
    ),
    (
        kimm_models.EfficientNetV2S.__name__,
        kimm_models.EfficientNetV2S,
        300,
        [
            ("STEM_S2", [1, 150, 150, make_divisible(24 * 1.0)]),
            ("BLOCK1_S4", [1, 75, 75, make_divisible(48 * 1.0)]),
            ("BLOCK2_S8", [1, 38, 38, make_divisible(64 * 1.0)]),
            ("BLOCK3_S16", [1, 19, 19, make_divisible(128 * 1.0)]),
            ("BLOCK5_S32", [1, 10, 10, make_divisible(256 * 1.0)]),
        ],
    ),
    (
        kimm_models.EfficientNetV2B0.__name__,
        kimm_models.EfficientNetV2B0,
        192,
        [
            ("STEM_S2", [1, 96, 96, make_divisible(32 * 1.0)]),
            ("BLOCK1_S4", [1, 48, 48, make_divisible(32 * 1.0)]),
            ("BLOCK2_S8", [1, 24, 24, make_divisible(48 * 1.0)]),
            ("BLOCK3_S16", [1, 12, 12, make_divisible(96 * 1.0)]),
            ("BLOCK5_S32", [1, 6, 6, make_divisible(192 * 1.0)]),
        ],
    ),
    (
        kimm_models.TinyNetE.__name__,
        kimm_models.TinyNetE,
        106,
        [
            ("STEM_S2", [1, 53, 53, 32]),
            ("BLOCK1_S4", [1, 27, 27, make_divisible(24 * 0.51)]),
            ("BLOCK2_S8", [1, 14, 14, make_divisible(40 * 0.51)]),
            ("BLOCK3_S16", [1, 7, 7, make_divisible(80 * 0.51)]),
            ("BLOCK5_S32", [1, 4, 4, make_divisible(192 * 0.51)]),
        ],
    ),
    # ghostnet
    (
        kimm_models.GhostNet100.__name__,
        kimm_models.GhostNet100,
        224,
        [
            ("STEM_S2", [1, 112, 112, 16]),
            ("BLOCK1_S4", [1, 56, 56, 24]),
            ("BLOCK3_S8", [1, 28, 28, 40]),
            ("BLOCK5_S16", [1, 14, 14, 80]),
            ("BLOCK7_S32", [1, 7, 7, 160]),
        ],
    ),
    (
        kimm_models.GhostNet100V2.__name__,
        kimm_models.GhostNet100V2,
        224,
        [
            ("STEM_S2", [1, 112, 112, 16]),
            ("BLOCK1_S4", [1, 56, 56, 24]),
            ("BLOCK3_S8", [1, 28, 28, 40]),
            ("BLOCK5_S16", [1, 14, 14, 80]),
            ("BLOCK7_S32", [1, 7, 7, 160]),
        ],
    ),
    # inception_next
    (
        kimm_models.InceptionNeXtTiny.__name__,
        kimm_models.InceptionNeXtTiny,
        224,
        [
            ("STEM_S4", [1, 56, 56, 96]),
            ("BLOCK0_S4", [1, 56, 56, 96]),
            ("BLOCK1_S8", [1, 28, 28, 192]),
            ("BLOCK2_S16", [1, 14, 14, 384]),
            ("BLOCK3_S32", [1, 7, 7, 768]),
        ],
    ),
    # inception_v3
    (
        kimm_models.InceptionV3.__name__,
        kimm_models.InceptionV3,
        299,
        [
            ("STEM_S2", [1, 147, 147, 64]),
            ("BLOCK0_S4", [1, 71, 71, 192]),
            ("BLOCK1_S8", [1, 35, 35, 288]),
            ("BLOCK2_S16", [1, 17, 17, 768]),
            ("BLOCK3_S32", [1, 8, 8, 2048]),
        ],
    ),
    # mobilenet_v2
    (
        kimm_models.MobileNetV2W050.__name__,
        kimm_models.MobileNetV2W050,
        224,
        [
            ("STEM_S2", [1, 112, 112, make_divisible(32 * 0.5)]),
            ("BLOCK1_S4", [1, 56, 56, make_divisible(24 * 0.5)]),
            ("BLOCK2_S8", [1, 28, 28, make_divisible(32 * 0.5)]),
            ("BLOCK3_S16", [1, 14, 14, make_divisible(64 * 0.5)]),
            ("BLOCK5_S32", [1, 7, 7, make_divisible(160 * 0.5)]),
        ],
    ),
    # mobilenet_v3
    (
        kimm_models.LCNet100.__name__,
        kimm_models.LCNet100,
        224,
        [
            ("STEM_S2", [1, 112, 112, make_divisible(16 * 1.0)]),
            ("BLOCK1_S4", [1, 56, 56, make_divisible(64 * 1.0)]),
            ("BLOCK2_S8", [1, 28, 28, make_divisible(128 * 1.0)]),
            ("BLOCK3_S16", [1, 14, 14, make_divisible(256 * 1.0)]),
            ("BLOCK5_S32", [1, 7, 7, make_divisible(512 * 1.0)]),
        ],
    ),
    (
        kimm_models.MobileNetV3W050Small.__name__,
        kimm_models.MobileNetV3W050Small,
        224,
        [
            ("STEM_S2", [1, 112, 112, 16]),
            ("BLOCK0_S4", [1, 56, 56, 8]),
            ("BLOCK1_S8", [1, 28, 28, 16]),
            ("BLOCK2_S16", [1, 14, 14, 24]),
            ("BLOCK4_S32", [1, 7, 7, 48]),
        ],
    ),
    (
        kimm_models.MobileNetV3W100SmallMinimal.__name__,
        kimm_models.MobileNetV3W100SmallMinimal,
        224,
        [
            ("STEM_S2", [1, 112, 112, make_divisible(16 * 1.0)]),
            ("BLOCK0_S4", [1, 56, 56, make_divisible(16 * 1.0)]),
            ("BLOCK1_S8", [1, 28, 28, make_divisible(24 * 1.0)]),
            ("BLOCK2_S16", [1, 14, 14, make_divisible(40 * 1.0)]),
            ("BLOCK4_S32", [1, 7, 7, make_divisible(96 * 1.0)]),
        ],
    ),
    # mobileone
    (
        kimm_models.MobileOneS0.__name__,
        kimm_models.MobileOneS0,
        224,
        [
            ("STEM_S2", [1, 112, 112, 48]),
            ("BLOCK0_S4", [1, 56, 56, 48]),
            ("BLOCK1_S8", [1, 28, 28, 128]),
            ("BLOCK2_S16", [1, 14, 14, 256]),
            ("BLOCK3_S32", [1, 7, 7, 1024]),
        ],
    ),
    # mobilevit
    (
        kimm_models.MobileViTS.__name__,
        kimm_models.MobileViTS,
        256,
        [
            ("STEM_S2", [1, 128, 128, 16]),
            ("BLOCK1_S4", [1, 64, 64, 64]),
            ("BLOCK2_S8", [1, 32, 32, 96]),
            ("BLOCK3_S16", [1, 16, 16, 128]),
            ("BLOCK4_S32", [1, 8, 8, 160]),
        ],
    ),
    # mobilevitv2
    (
        kimm_models.MobileViTV2W050.__name__,
        kimm_models.MobileViTV2W050,
        256,
        [
            ("STEM_S2", [1, 128, 128, 16]),
            ("BLOCK1_S4", [1, 64, 64, 64]),
            ("BLOCK2_S8", [1, 32, 32, 128]),
            ("BLOCK3_S16", [1, 16, 16, 192]),
            ("BLOCK4_S32", [1, 8, 8, 256]),
        ],
    ),
    # regnet
    (
        kimm_models.RegNetX002.__name__,
        kimm_models.RegNetX002,
        224,
        [
            ("STEM_S2", [1, 112, 112, 32]),
            ("BLOCK0_S4", [1, 56, 56, 24]),
            ("BLOCK1_S8", [1, 28, 28, 56]),
            ("BLOCK2_S16", [1, 14, 14, 152]),
            ("BLOCK3_S32", [1, 7, 7, 368]),
        ],
    ),
    (
        kimm_models.RegNetY002.__name__,
        kimm_models.RegNetY002,
        224,
        [
            ("STEM_S2", [1, 112, 112, 32]),
            ("BLOCK0_S4", [1, 56, 56, 24]),
            ("BLOCK1_S8", [1, 28, 28, 56]),
            ("BLOCK2_S16", [1, 14, 14, 152]),
            ("BLOCK3_S32", [1, 7, 7, 368]),
        ],
    ),
    # repvgg
    (
        kimm_models.RepVGGA0.__name__,
        kimm_models.RepVGGA0,
        224,
        [
            ("STEM_S2", [1, 112, 112, 48]),
            ("BLOCK0_S4", [1, 56, 56, 48]),
            ("BLOCK1_S8", [1, 28, 28, 96]),
            ("BLOCK2_S16", [1, 14, 14, 192]),
            ("BLOCK3_S32", [1, 7, 7, 1280]),
        ],
    ),
    # resnet
    (
        kimm_models.ResNet18.__name__,
        kimm_models.ResNet18,
        224,
        [
            ("STEM_S2", [1, 112, 112, 64]),
            ("BLOCK0_S4", [1, 56, 56, 64]),
            ("BLOCK1_S8", [1, 28, 28, 128]),
            ("BLOCK2_S16", [1, 14, 14, 256]),
            ("BLOCK3_S32", [1, 7, 7, 512]),
        ],
    ),
    (
        kimm_models.ResNet50.__name__,
        kimm_models.ResNet50,
        224,
        [
            ("STEM_S2", [1, 112, 112, 64]),
            ("BLOCK0_S4", [1, 56, 56, 64 * 4]),
            ("BLOCK1_S8", [1, 28, 28, 128 * 4]),
            ("BLOCK2_S16", [1, 14, 14, 256 * 4]),
            ("BLOCK3_S32", [1, 7, 7, 512 * 4]),
        ],
    ),
    # vgg
    (
        kimm_models.VGG11.__name__,
        kimm_models.VGG11,
        224,
        [
            ("BLOCK0_S1", [1, 224, 224, 64]),
            ("BLOCK1_S2", [1, 112, 112, 128]),
            ("BLOCK2_S4", [1, 56, 56, 256]),
            ("BLOCK3_S8", [1, 28, 28, 512]),
            ("BLOCK4_S16", [1, 14, 14, 512]),
            ("BLOCK5_S32", [1, 7, 7, 512]),
        ],
        None,  # skip weights to save time
    ),
    # vision_transformer
    (
        kimm_models.VisionTransformerTiny16.__name__,
        kimm_models.VisionTransformerTiny16,
        384,
        [*((f"BLOCK{i}", [1, 577, 192]) for i in range(5))],
    ),
    (
        kimm_models.VisionTransformerTiny32.__name__,
        kimm_models.VisionTransformerTiny32,
        384,
        [*((f"BLOCK{i}", [1, 145, 192]) for i in range(5))],
        None,  # no weights
    ),
    # xception
    (
        kimm_models.Xception.__name__,
        kimm_models.Xception,
        299,
        [
            ("STEM_S2", [1, 147, 147, 64]),
            ("BLOCK0_S4", [1, 74, 74, 128]),
            ("BLOCK1_S8", [1, 37, 37, 256]),
            ("BLOCK2_S16", [1, 19, 19, 728]),
            ("BLOCK3_S32", [1, 10, 10, 2048]),
        ],
    ),
]


@pytest.mark.requires_trainable_backend  # numpy is too slow to test
class ModelTest(testing.TestCase, parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_image_data_format = backend.image_data_format()

    @classmethod
    def tearDownClass(cls):
        backend.set_image_data_format(cls.original_image_data_format)

    @parameterized.named_parameters(MODEL_CONFIGS)
    def test_model_base_channels_last(
        self, model_class, image_size, features, weights="imagenet"
    ):
        backend.set_image_data_format("channels_last")
        model = model_class(weights=weights)
        image_path = keras.utils.get_file(
            "african_elephant.jpg", "https://i.imgur.com/Bvro0YD.png"
        )
        # preprocessing
        image = utils.load_img(image_path, target_size=(image_size, image_size))
        image = utils.img_to_array(image, data_format="channels_last")
        x = ops.convert_to_tensor(image)
        x = ops.expand_dims(x, axis=0)

        y = model(x, training=False)

        if weights == "imagenet":
            names = [p[1] for p in decode_predictions(y)[0]]
            # Test correct label is in top 3 (weak correctness test).
            self.assertIn("African_elephant", names[:3])
        elif weights is None:
            self.assertEqual(list(y.shape), [1, 1000])

    @parameterized.named_parameters(MODEL_CONFIGS)
    def test_model_base_channels_first(
        self, model_class, image_size, features, weights="imagenet"
    ):
        if (
            len(tf.config.list_physical_devices("GPU")) == 0
            and backend.backend() == "tensorflow"
        ):
            self.skipTest(
                "Conv2D doesn't support channels_first using CPU with "
                "tensorflow backend"
            )

        backend.set_image_data_format("channels_first")
        model = model_class(weights=weights)
        image_path = keras.utils.get_file(
            "african_elephant.jpg", "https://i.imgur.com/Bvro0YD.png"
        )
        # preprocessing
        image = utils.load_img(image_path, target_size=(image_size, image_size))
        image = utils.img_to_array(image, data_format="channels_first")
        x = ops.convert_to_tensor(image)
        x = ops.expand_dims(x, axis=0)

        y = model(x, training=False)

        if weights == "imagenet":
            names = [p[1] for p in decode_predictions(y)[0]]
            # Test correct label is in top 3 (weak correctness test).
            self.assertIn("African_elephant", names[:3])
        elif weights is None:
            self.assertEqual(list(y.shape), [1, 1000])

    @parameterized.named_parameters(MODEL_CONFIGS)
    def test_model_feature_extractor(
        self, model_class, image_size, features, weights="imagenet"
    ):
        backend.set_image_data_format("channels_last")
        x = random.uniform([1, image_size, image_size, 3]) * 255.0
        model = model_class(weights=None, feature_extractor=True)

        y = model(x, training=False)

        self.assertIsInstance(y, dict)
        self.assertContainsSubset(
            model_class.available_feature_keys, list(y.keys())
        )
        for feature_info in features:
            name, shape = feature_info
            self.assertEqual(list(y[name].shape), shape)

    @parameterized.named_parameters(
        (kimm_models.RepVGGA0.__name__, kimm_models.RepVGGA0, 224),
        (kimm_models.MobileOneS0.__name__, kimm_models.MobileOneS0, 224),
    )
    def test_model_get_reparameterized_model(self, model_class, image_size):
        x = random.uniform([1, image_size, image_size, 3]) * 255.0
        model = model_class()
        reparameterized_model = model.get_reparameterized_model()

        y1 = model(x, training=False)
        y2 = reparameterized_model(x, training=False)

        self.assertAllClose(y1, y2, atol=1e-5)

    @pytest.mark.serialization
    @parameterized.named_parameters(MODEL_CONFIGS)
    def test_model_serialization(
        self, model_class, image_size, features, weights="imagenet"
    ):
        backend.set_image_data_format("channels_last")
        x = random.uniform([1, image_size, image_size, 3]) * 255.0
        temp_dir = self.get_temp_dir()
        model1 = model_class(weights=None)

        y1 = model1(x, training=False)
        model1.save(temp_dir + "/model.keras")
        model2 = models.load_model(temp_dir + "/model.keras")
        y2 = model2(x, training=False)

        self.assertAllClose(y1, y2)

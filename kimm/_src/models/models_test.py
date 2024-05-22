import keras
import pytest
import tensorflow as tf
from absl.testing import parameterized
from keras.src import testing

from kimm._src import models as kimm_models
from kimm._src.utils.make_divisble import make_divisible
from kimm._src.utils.model_utils import get_reparameterized_model

decode_predictions = keras.applications.imagenet_utils.decode_predictions

# Test BaseModel


class SampleModel(kimm_models.BaseModel):
    available_feature_keys = [f"S{2**i}" for i in range(1, 4)]

    def __init__(self, **kwargs):
        self.set_properties(kwargs)
        inputs = keras.layers.Input(shape=[224, 224, 3])
        features = {}
        s2 = keras.layers.Conv2D(3, 1, 2, use_bias=False)(inputs)
        features["S2"] = s2
        s4 = keras.layers.Conv2D(3, 1, 2, use_bias=False)(s2)
        features["S4"] = s4
        s8 = keras.layers.Conv2D(3, 1, 2, use_bias=False)(s4)
        features["S8"] = s8
        outputs = keras.layers.GlobalAveragePooling2D()(s8)
        super().__init__(
            inputs=inputs, outputs=outputs, features=features, **kwargs
        )


class SampleModelStaticShape(kimm_models.BaseModel):
    def __init__(self, input_shape=None, input_tensor=None):
        self.determine_input_tensor(
            input_tensor, input_shape, static_shape=True
        )


class BaseModelTest(testing.TestCase, parameterized.TestCase):
    def test_feature_extractor(self):
        x = keras.random.uniform([1, 224, 224, 3])

        # Test availiable_feature_keys
        self.assertContainsSubset(
            ["S2", "S4", "S8"], SampleModel.available_feature_keys
        )

        # Test feature_extractor=False
        model = SampleModel()
        y = model(x, training=False)
        self.assertNotIsInstance(y, dict)
        self.assertEqual(list(y.shape), [1, 3])

        # Test feature_extractor=True
        model = SampleModel(feature_extractor=True)
        y = model(x, training=False)
        self.assertIsInstance(y, dict)
        self.assertEqual(list(y["S2"].shape), [1, 112, 112, 3])
        self.assertEqual(list(y["S8"].shape), [1, 28, 28, 3])

        # Test feature_extractor=True with feature_keys
        model = SampleModel(
            include_top=False, feature_extractor=True, feature_keys=["S2", "S8"]
        )
        y = model(x, training=False)
        self.assertIsInstance(y, dict)
        self.assertNotIn("S4", y)
        self.assertEqual(list(y["S2"].shape), [1, 112, 112, 3])
        self.assertEqual(list(y["S8"].shape), [1, 28, 28, 3])
        self.assertNotIn("TOP", y)

    def test_determine_input_tensor(self):
        # Test static shape
        SampleModelStaticShape(input_shape=[224, 224, 3])
        inputs = keras.Input([224, 224, 3])
        SampleModelStaticShape(input_tensor=inputs)

        # Test arbitrary shape
        with self.assertRaisesRegex(ValueError, "The inferred input_shape"):
            SampleModelStaticShape(input_shape=[None, None, 3])
        with self.assertRaisesRegex(ValueError, "The inferred input_shape"):
            inputs = keras.Input([None, None, 3])
            SampleModelStaticShape(input_tensor=inputs)


# Test some small models

# name, class, default_size, features (name, shape),
# weights (defaults to imagenet)
MODEL_CONFIGS = [
    # convmixer
    (
        kimm_models.convmixer.ConvMixer736D32.__name__,
        kimm_models.convmixer.ConvMixer736D32,
        224,
        [
            ("STEM", [1, 32, 32, 768]),
            *((f"BLOCK{i}", [1, 32, 32, 768]) for i in range(32)),
        ],
    ),
    # convnext
    (
        kimm_models.convnext.ConvNeXtAtto.__name__,
        kimm_models.convnext.ConvNeXtAtto,
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
        kimm_models.densenet.DenseNet121.__name__,
        kimm_models.densenet.DenseNet121,
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
        kimm_models.efficientnet.EfficientNetB2.__name__,
        kimm_models.efficientnet.EfficientNetB2,
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
        kimm_models.efficientnet.EfficientNetLiteB2.__name__,
        kimm_models.efficientnet.EfficientNetLiteB2,
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
        kimm_models.efficientnet.EfficientNetV2S.__name__,
        kimm_models.efficientnet.EfficientNetV2S,
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
        kimm_models.efficientnet.EfficientNetV2B0.__name__,
        kimm_models.efficientnet.EfficientNetV2B0,
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
        kimm_models.efficientnet.TinyNetE.__name__,
        kimm_models.efficientnet.TinyNetE,
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
        kimm_models.ghostnet.GhostNet100.__name__,
        kimm_models.ghostnet.GhostNet100,
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
        kimm_models.ghostnet.GhostNet100V2.__name__,
        kimm_models.ghostnet.GhostNet100V2,
        224,
        [
            ("STEM_S2", [1, 112, 112, 16]),
            ("BLOCK1_S4", [1, 56, 56, 24]),
            ("BLOCK3_S8", [1, 28, 28, 40]),
            ("BLOCK5_S16", [1, 14, 14, 80]),
            ("BLOCK7_S32", [1, 7, 7, 160]),
        ],
    ),
    # hgnet
    (
        kimm_models.hgnet.HGNetTiny.__name__,
        kimm_models.hgnet.HGNetTiny,
        224,
        [
            ("STEM_S4", [1, 56, 56, 96]),
            ("BLOCK0_S4", [1, 56, 56, 224]),
            ("BLOCK1_S8", [1, 28, 28, 448]),
            ("BLOCK2_S16", [1, 14, 14, 512]),
            ("BLOCK3_S32", [1, 7, 7, 768]),
        ],
    ),
    (
        kimm_models.hgnet.HGNetV2B0.__name__,
        kimm_models.hgnet.HGNetV2B0,
        224,
        [
            ("STEM_S4", [1, 56, 56, 16]),
            ("BLOCK0_S4", [1, 56, 56, 64]),
            ("BLOCK1_S8", [1, 28, 28, 256]),
            ("BLOCK2_S16", [1, 14, 14, 512]),
            ("BLOCK3_S32", [1, 7, 7, 1024]),
        ],
    ),
    # inception_next
    (
        kimm_models.inception_next.InceptionNeXtTiny.__name__,
        kimm_models.inception_next.InceptionNeXtTiny,
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
        kimm_models.inception_v3.InceptionV3.__name__,
        kimm_models.inception_v3.InceptionV3,
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
        kimm_models.mobilenet_v2.MobileNetV2W050.__name__,
        kimm_models.mobilenet_v2.MobileNetV2W050,
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
        kimm_models.mobilenet_v3.LCNet100.__name__,
        kimm_models.mobilenet_v3.LCNet100,
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
        kimm_models.mobilenet_v3.MobileNetV3W050Small.__name__,
        kimm_models.mobilenet_v3.MobileNetV3W050Small,
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
        kimm_models.mobilenet_v3.MobileNetV3W100SmallMinimal.__name__,
        kimm_models.mobilenet_v3.MobileNetV3W100SmallMinimal,
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
        kimm_models.mobileone.MobileOneS0.__name__,
        kimm_models.mobileone.MobileOneS0,
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
        kimm_models.mobilevit.MobileViTS.__name__,
        kimm_models.mobilevit.MobileViTS,
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
        kimm_models.mobilevit.MobileViTV2W050.__name__,
        kimm_models.mobilevit.MobileViTV2W050,
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
        kimm_models.regnet.RegNetX002.__name__,
        kimm_models.regnet.RegNetX002,
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
        kimm_models.regnet.RegNetY002.__name__,
        kimm_models.regnet.RegNetY002,
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
        kimm_models.repvgg.RepVGGA0.__name__,
        kimm_models.repvgg.RepVGGA0,
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
        kimm_models.resnet.ResNet18.__name__,
        kimm_models.resnet.ResNet18,
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
        kimm_models.resnet.ResNet50.__name__,
        kimm_models.resnet.ResNet50,
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
        kimm_models.vgg.VGG11.__name__,
        kimm_models.vgg.VGG11,
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
        kimm_models.vision_transformer.VisionTransformerTiny16.__name__,
        kimm_models.vision_transformer.VisionTransformerTiny16,
        384,
        [*((f"BLOCK{i}", [1, 577, 192]) for i in range(5))],
    ),
    (
        kimm_models.vision_transformer.VisionTransformerTiny32.__name__,
        kimm_models.vision_transformer.VisionTransformerTiny32,
        384,
        [*((f"BLOCK{i}", [1, 145, 192]) for i in range(5))],
        None,  # no weights
    ),
    # xception
    (
        kimm_models.xception.Xception.__name__,
        kimm_models.xception.Xception,
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
class ModelsTest(testing.TestCase, parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_image_data_format = keras.backend.image_data_format()

    @classmethod
    def tearDownClass(cls):
        keras.backend.set_image_data_format(cls.original_image_data_format)

    @parameterized.named_parameters(MODEL_CONFIGS)
    def test_predict(
        self,
        model_class,
        image_size,
        features,
        weights="imagenet",
    ):
        # We also enable feature_extractor=True in model instantiation to
        # speed up the testing

        # Load the image
        image_path = keras.utils.get_file(
            "elephant.png",
            "https://github.com/james77777778/keras-image-models/releases/download/0.1.0/elephant.png",
        )
        image = keras.utils.load_img(
            image_path, target_size=(image_size, image_size)
        )

        # Test channels_last and feature_extractor=True
        keras.backend.set_image_data_format("channels_last")
        model = model_class(weights=weights, feature_extractor=True)
        x = keras.utils.img_to_array(image, data_format="channels_last")
        x = keras.ops.expand_dims(keras.ops.convert_to_tensor(x), axis=0)

        y = model(x, training=False)

        # Verify output correctness
        prob = y["TOP"]
        if weights == "imagenet":
            names = [p[1] for p in decode_predictions(prob)[0]]
            # Test correct label is in top 3 (weak correctness test).
            self.assertIn("African_elephant", names[:3])
        elif weights is None:
            self.assertEqual(list(prob.shape), [1, 1000])

        # Verify features
        self.assertIsInstance(y, dict)
        self.assertContainsSubset(
            model_class.available_feature_keys, list(y.keys())
        )
        for feature_info in features:
            name, shape = feature_info
            self.assertEqual(list(y[name].shape), shape)

        # Test channels_first
        if (
            len(tf.config.list_physical_devices("GPU")) == 0
            and keras.backend.backend() == "tensorflow"
        ):
            # TensorFlow doesn't support channels_first using CPU
            return

        keras.backend.set_image_data_format("channels_first")
        model = model_class(weights=weights)
        x = keras.utils.img_to_array(image, data_format="channels_first")
        x = keras.ops.expand_dims(keras.ops.convert_to_tensor(x), axis=0)

        y = model(x, training=False)

        # Verify output correctness
        if weights == "imagenet":
            names = [p[1] for p in decode_predictions(y)[0]]
            # Test correct label is in top 3 (weak correctness test).
            self.assertIn("African_elephant", names[:3])
        elif weights is None:
            self.assertEqual(list(y.shape), [1, 1000])

    @parameterized.named_parameters(
        (
            kimm_models.repvgg.RepVGGA0.__name__,
            kimm_models.repvgg.RepVGGA0,
            224,
        ),
        (
            kimm_models.mobileone.MobileOneS0.__name__,
            kimm_models.mobileone.MobileOneS0,
            224,
        ),
    )
    def test_get_reparameterized_model(self, model_class, image_size):
        x = keras.random.uniform([1, image_size, image_size, 3], seed=2024)
        model = model_class(weights=None)
        reparameterized_model = get_reparameterized_model(model)

        y1 = model(x, training=False)
        y2 = reparameterized_model(x, training=False)

        self.assertAllClose(y1, y2, atol=1e-1)  # CPU: atol=1e-5

    @parameterized.named_parameters(
        (
            kimm_models.mobilevit.MobileViTXXS.__name__,
            kimm_models.mobilevit.MobileViTXXS,
            224,  # default_size=256
        ),
        (
            kimm_models.mobilevit.MobileViTV2W050.__name__,
            kimm_models.mobilevit.MobileViTV2W050,
            224,  # default_size=256
        ),
        (
            kimm_models.vision_transformer.VisionTransformerTiny16.__name__,
            kimm_models.vision_transformer.VisionTransformerTiny16,
            224,  # default_size=384
        ),
    )
    def test_arbitrary_shape(self, model_class, image_size):
        x = keras.random.uniform([1, image_size, image_size, 3]) * 255.0
        model = model_class(
            input_shape=[image_size, image_size, 3], weights=None
        )
        model(x, training=False)

    @pytest.mark.serialization
    @parameterized.named_parameters(MODEL_CONFIGS)
    def test_serialization(
        self,
        model_class,
        image_size,
        features,
        weights="imagenet",
    ):
        keras.backend.set_image_data_format("channels_last")
        x = keras.random.uniform([1, image_size, image_size, 3]) * 255.0
        temp_dir = self.get_temp_dir()
        model1 = model_class(weights=None)
        if hasattr(model1, "get_reparameterized_model"):
            model1 = model1.get_reparameterized_model()

        y1 = model1(x, training=False)
        model1.save(temp_dir + "/model.keras")
        model2 = keras.models.load_model(temp_dir + "/model.keras")
        y2 = model2(x, training=False)

        self.assertAllClose(y1, y2)

import pathlib
import typing
import urllib.parse

from keras import KerasTensor
from keras import backend
from keras import layers
from keras import models
from keras import utils
from keras.src.applications import imagenet_utils


class BaseModel(models.Model):
    default_origin = (
        "https://github.com/james77777778/kimm/releases/download/0.1.0/"
    )
    available_feature_keys = []
    available_weights = []

    def __init__(
        self,
        inputs,
        outputs,
        features: typing.Optional[typing.Dict[str, KerasTensor]] = None,
        **kwargs,
    ):
        if not hasattr(self, "_feature_extractor"):
            del features
            super().__init__(inputs=inputs, outputs=outputs, **kwargs)
        else:
            if not hasattr(self, "_feature_keys"):
                raise AttributeError(
                    "`self._feature_keys` must be set when initializing "
                    "BaseModel"
                )
            if self._feature_extractor:
                if features is None:
                    raise ValueError(
                        "`features` must be set when `feature_extractor=True`. "
                        f"Received features={features}"
                    )
                if self._feature_keys is None:
                    self._feature_keys = list(features.keys())
                filtered_features = {}
                for k in self._feature_keys:
                    if k not in features:
                        raise KeyError(
                            f"'{k}' is not a key of `features`. Available keys "
                            f"are: {list(features.keys())}"
                        )
                    filtered_features[k] = features[k]
                # Add outputs
                if backend.is_keras_tensor(outputs):
                    filtered_features["TOP"] = outputs
                super().__init__(
                    inputs=inputs, outputs=filtered_features, **kwargs
                )
            else:
                del features
                super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        if hasattr(self, "_weights_url"):
            self.load_pretrained_weights(self._weights_url)

    def set_properties(
        self, kwargs: typing.Dict[str, typing.Any], default_size: int = 224
    ):
        """Must be called in the initilization of the class.

        This method will add following common properties to the model object:
        - input_shape
        - include_preprocessing
        - include_top
        - pooling
        - dropout_rate
        - classes
        - classifier_activation
        - _weights
        - weights_url
        - default_size
        """
        self._input_shape = kwargs.pop("input_shape", None)
        self._include_preprocessing = kwargs.pop("include_preprocessing", True)
        self._include_top = kwargs.pop("include_top", True)
        self._pooling = kwargs.pop("pooling", None)
        self._dropout_rate = kwargs.pop("dropout_rate", 0.0)
        self._classes = kwargs.pop("classes", 1000)
        self._classifier_activation = kwargs.pop(
            "classifier_activation", "softmax"
        )
        self._weights = kwargs.pop("weights", None)
        self._weights_url = kwargs.pop("weights_url", None)
        self._default_size = kwargs.pop("default_size", default_size)
        # feature extractor
        self._feature_extractor = kwargs.pop("feature_extractor", False)
        self._feature_keys = kwargs.pop("feature_keys", None)

    def determine_input_tensor(
        self,
        input_tensor: typing.Optional[KerasTensor] = None,
        input_shape: typing.Optional[typing.Sequence[int]] = None,
        default_size: int = 224,
        min_size: int = 32,
        require_flatten: bool = False,
        static_shape: bool = False,
    ):
        """Determine the input tensor by the arguments."""
        input_shape = imagenet_utils.obtain_input_shape(
            input_shape,
            default_size=default_size,
            min_size=min_size,
            data_format=backend.image_data_format(),
            require_flatten=require_flatten or static_shape,
            weights=None,
        )

        if input_tensor is None:
            x = layers.Input(shape=input_shape)
        else:
            if not backend.is_keras_tensor(input_tensor):
                x = layers.Input(tensor=input_tensor, shape=input_shape)
            else:
                x = utils.get_source_inputs(input_tensor)
        return x

    def build_preprocessing(
        self,
        inputs,
        mode: typing.Literal["imagenet", "0_1", "-1_1"] = "imagenet",
    ):
        if self._include_preprocessing is False:
            return inputs
        channels_axis = (
            -1 if backend.image_data_format() == "channels_last" else -3
        )
        if mode == "imagenet":
            # [0, 255] to [0, 1] and apply ImageNet mean and variance
            x = layers.Rescaling(scale=1.0 / 255.0)(inputs)
            x = layers.Normalization(
                axis=channels_axis,
                mean=[0.485, 0.456, 0.406],
                variance=[0.229, 0.224, 0.225],
            )(x)
        elif mode == "0_1":
            # [0, 255] to [-1, 1]
            x = layers.Rescaling(scale=1.0 / 255.0)(inputs)
        elif mode == "-1_1":
            # [0, 255] to [-1, 1]
            x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)(inputs)
        else:
            raise ValueError(
                "`mode` must be one of ('imagenet', '0_1', '-1_1'). "
                f"Received: mode={mode}"
            )
        return x

    def build_top(
        self,
        inputs,
        classes: int,
        classifier_activation: str,
        dropout_rate: float,
    ):
        x = layers.GlobalAveragePooling2D(name="avg_pool")(inputs)
        x = layers.Dropout(rate=dropout_rate, name="head_dropout")(x)
        x = layers.Dense(
            classes, activation=classifier_activation, name="classifier"
        )(x)
        return x

    def build_head(self, inputs):
        x = inputs
        if self._include_top:
            x = self.build_top(
                x,
                self._classes,
                self._classifier_activation,
                self._dropout_rate,
            )
        else:
            if self._pooling == "avg":
                x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            elif self._pooling == "max":
                x = layers.GlobalMaxPooling2D(name="max_pool")(x)
        return x

    def load_pretrained_weights(self, weights_url: typing.Optional[str] = None):
        if weights_url is not None:
            result = urllib.parse.urlparse(weights_url)
            file_name = pathlib.Path(result.path).name
            weights_path = utils.get_file(
                file_name, weights_url, cache_subdir="kimm_models"
            )
            self.load_weights(weights_path)

    def get_config(self):
        # Don't chain to super here. The default `get_config()` for functional
        # models is nested and cannot be passed to BaseModel.
        config = {
            # models.Model
            "name": self.name,
            "trainable": self.trainable,
            "input_shape": self.input_shape[1:],
            # common
            "include_preprocessing": self._include_preprocessing,
            "include_top": self._include_top,
            "pooling": self._pooling,
            "dropout_rate": self._dropout_rate,
            "classes": self._classes,
            "classifier_activation": self._classifier_activation,
            "weights": self._weights,
            "weights_url": self._weights_url,
            # feature extractor
            "feature_extractor": self._feature_extractor,
            "feature_keys": self._feature_keys,
        }
        return config

    def fix_config(self, config: typing.Dict):
        return config

    def get_weights_url(self, weights):
        if weights is None:
            return None

        for _weights, _origin, _file_name in self.available_weights:
            if weights == _weights:
                return f"{_origin}/{_file_name}"

        # Failed to find the weights
        _available_weights_name = [
            _weights for _weights, _ in self.available_weights
        ]
        raise ValueError(
            f"Available weights are {_available_weights_name}. "
            f"Received weights={weights}"
        )

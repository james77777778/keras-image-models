import pathlib
import typing
import urllib.parse

from keras import KerasTensor
from keras import backend
from keras import layers
from keras import models
from keras import utils
from keras.src.applications import imagenet_utils

from kimm._src.kimm_export import kimm_export


@kimm_export(parent_path=["kimm.models", "kimm.models.base_model"])
class BaseModel(models.Model):
    default_origin = "https://github.com/james77777778/keras-image-models/releases/download/0.1.0/"
    available_feature_keys = []
    available_weights = []

    def __new__(cls, *args, **kwargs):
        # Fix type hint
        return typing.cast(BaseModel, super().__new__(cls, *args, **kwargs))

    def __init__(
        self,
        inputs,
        outputs,
        features: typing.Optional[typing.Dict[str, KerasTensor]] = None,
        **kwargs,
    ):
        self._check_feature_extractor_setting(features)
        _include_top = getattr(self, "_include_top", True)
        if self._feature_extractor:
            if self._feature_keys is None:
                self._feature_keys = list(features.keys())
            filtered_features = {}
            for k in self._feature_keys:
                filtered_features[k] = features[k]
            # Add outputs
            if _include_top and backend.is_keras_tensor(outputs):
                filtered_features["TOP"] = outputs
            super().__init__(inputs=inputs, outputs=filtered_features, **kwargs)
        else:
            del features
            super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        if hasattr(self, "_weights"):
            self._load_pretrained_weights(self._weights)

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
        - weights
        - default_size
        - feature_extractor
        - feature_keys
        """
        # Common properties
        self._input_shape = kwargs.pop("input_shape", None)
        self._include_preprocessing = kwargs.pop("include_preprocessing", True)
        self._include_top = kwargs.pop("include_top", True)
        self._pooling = kwargs.pop("pooling", None)
        self._dropout_rate = kwargs.pop("dropout_rate", 0.0)
        self._classes = kwargs.pop("classes", 1000)
        self._classifier_activation = kwargs.pop(
            "classifier_activation", "softmax"
        )
        self._weights = self._parse_weights(kwargs.pop("weights", None))
        self._default_size = int(kwargs.pop("default_size", default_size))
        # For feature extractor
        self._feature_extractor = bool(kwargs.pop("feature_extractor", False))
        self._feature_keys = kwargs.pop("feature_keys", None)
        # Internal parameters
        self._preprocessing_mode = False

    @property
    def input_shape(self):
        """The input shape of the model.

        `None` means that the dimension can be of arbitrary size.

        Note: Some models, especially those including attention-related layers,
        require a static shape.
        """
        return tuple(super().input_shape)

    @property
    def default_size(self):
        """The default size of the model.

        `default_size` indicates the size of the inputs for the pretrained
        model. For example, when loading "imagenet" weights, you should get
        good results by feeding the inputs with this size.
        """
        return self._default_size

    @property
    def preprocessing_mode(self):
        """The mode of the preprocessing of the model.

        - `False`: No preprocessing.
        - `"imagenet"`: Scale the value range from [0, 255] to [0, 1], then
            normalize with the mean `(0.485, 0.456, 0.406)` and variance
            `(0.229, 0.224, 0.225)`.
        - `"0_1"`: Scale the value range from [0, 255] to [0, 1].
        - `"-1_1"`: Scale the value range from [0, 255] to [-1, 1].
        """
        return self._preprocessing_mode

    @property
    def feature_extractor(self):
        """Whether this model is a feature extractor."""
        return self._feature_extractor

    @property
    def feature_keys(self):
        """The keys of the features if the model is a feature extractor."""
        return self._feature_keys

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
        try:
            input_shape = imagenet_utils.obtain_input_shape(
                input_shape,
                default_size=default_size,
                min_size=min_size,
                data_format=backend.image_data_format(),
                require_flatten=require_flatten or static_shape,
                weights=None,
            )
        except ValueError as e:
            # Override the error msg from `obtain_input_shape`
            if "If `include_top` is True" in str(e):
                raise ValueError(
                    f"The inferred input_shape={input_shape} must be "
                    f"static for {self.__class__.__name__} "
                    f"(require_flatten={require_flatten}, "
                    f"static_shape={static_shape})."
                ) from None
        if input_tensor is None:
            x = layers.Input(shape=input_shape)
        else:
            if not backend.is_keras_tensor(input_tensor):
                x = layers.Input(tensor=input_tensor, shape=input_shape)
            else:
                x = utils.get_source_inputs(input_tensor)[0]
        if static_shape:
            if None in x.shape[1:]:
                raise ValueError(
                    f"The inferred input_shape={x.shape} must be "
                    f"static for {self.__class__.__name__} "
                    f"(require_flatten={require_flatten}, "
                    f"static_shape={static_shape})."
                )
        return x

    def build_preprocessing(
        self,
        inputs,
        mode: typing.Literal["imagenet", "0_1", "-1_1"] = "imagenet",
    ):
        """Build the preprocessing pipeline.

        Args:
            inputs: A `KerasTensor` indicating the input.
            mode: A `str` indicating the preprocessing mode. The available modes
                are {"imagenet", "0_1", "-1_1"}. Defaults to "imagenet".

        Returns:
            A `KerasTensor`.
        """
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
            # [0, 255] to [0, 1]
            x = layers.Rescaling(scale=1.0 / 255.0)(inputs)
        elif mode == "-1_1":
            # [0, 255] to [-1, 1]
            x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)(inputs)
        else:
            raise ValueError(
                "`mode` must be one of ('imagenet', '0_1', '-1_1'). "
                f"Received: mode={mode}"
            )
        self._preprocessing_mode = mode
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
            # feature extractor
            "feature_extractor": self._feature_extractor,
            "feature_keys": self._feature_keys,
        }
        return config

    def fix_config(self, config: typing.Dict):
        return config

    def __repr__(self):
        repr_str = (
            f"<{self.__class__.__name__} "
            f"name={self.name}, "
            f"input_shape={self.input_shape}, "
            f"default_size={self._default_size}, "
        )
        preprocessing_mode = self._preprocessing_mode
        if preprocessing_mode is not False:
            preprocessing_mode = f'"{self._preprocessing_mode}"'
        repr_str += f"preprocessing_mode={preprocessing_mode}, "
        repr_str += f"feature_extractor={self._feature_extractor}, "
        repr_str += f"feature_keys={self._feature_keys}"
        repr_str += ">"
        return repr_str

    def _check_feature_extractor_setting(self, features):
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
            _feature_keys = self._feature_keys
            if _feature_keys is None:
                _feature_keys = list(features.keys())
            for k in _feature_keys:
                if k not in features:
                    raise KeyError(
                        f"'{k}' is not a key of `features`. Available keys "
                        f"are: {list(features.keys())}"
                    )

    def _parse_weights(self, weights: typing.Union[str, None]):
        """Parse the path/URL if weights is specified.

        This function prefers the following order:
        1. Local file (a instance of pathlib.Path)
        2. URL (a str starting with 'https://')
        3. Available weights (mostly 'imagenet')
        """
        if weights is None:
            return None
        # Check local files
        elif pathlib.Path(weights).exists():
            return pathlib.Path(weights)
        # Check URL
        elif str(weights).startswith(("http://", "https://")):
            return str(weights)
        # Check avaiable weights (a plain string)
        else:
            # Parse the string for available weights
            for _weights, _origin, _file_name in self.available_weights:
                if weights == _weights:
                    return f"{_origin}{_file_name}"

        # Failed to find the weights
        _available_weights_name = [
            _weights for _weights, _, _ in self.available_weights
        ]
        raise ValueError(
            f"If `weights` is a URL string, the available weights are "
            f"{_available_weights_name}. Received: weights={weights}"
        )

    def _load_pretrained_weights(
        self,
        weights: typing.Union[pathlib.Path, str, None] = None,
    ):
        """Load the pretrained weights from `weights_url`.

        This function prefers the following order:
        1. Local file (a instance of pathlib.Path)
        2. URL (a str starting with 'https://')
        """
        if weights is None:
            return
        # Check local files
        if isinstance(weights, pathlib.Path):
            self.load_weights(weights)
        # Check URL
        else:
            result = urllib.parse.urlparse(weights)
            file_name = pathlib.Path(result.path).name
            weights_path = utils.get_file(
                file_name, weights, cache_subdir="kimm_models"
            )
            self.load_weights(weights_path)

import abc
import typing

from keras import KerasTensor
from keras import backend
from keras import layers
from keras import models
from keras.src.applications import imagenet_utils


class BaseModel(models.Model):
    def __init__(
        self,
        inputs,
        outputs,
        features: typing.Optional[typing.Dict[str, KerasTensor]] = None,
        feature_keys: typing.Optional[typing.List[str]] = None,
        **kwargs,
    ):
        self.as_feature_extractor = kwargs.pop("as_feature_extractor", False)
        self.feature_keys = feature_keys
        if self.as_feature_extractor:
            if features is None:
                raise ValueError(
                    "`features` must be set when "
                    f"`as_feature_extractor=True`. Got features={features}"
                )
            if self.feature_keys is None:
                self.feature_keys = list(features.keys())
            filtered_features = {}
            for k in self.feature_keys:
                if k not in features:
                    raise KeyError(
                        f"'{k}' is not a key of `features`. Available keys "
                        f"are: {list(features.keys())}"
                    )
                filtered_features[k] = features[k]
            super().__init__(inputs=inputs, outputs=filtered_features, **kwargs)
        else:
            del features
            super().__init__(inputs=inputs, outputs=outputs, **kwargs)

    def parse_kwargs(
        self, kwargs: typing.Dict[str, typing.Any], default_size: int = 224
    ):
        result = {
            "input_tensor": kwargs.pop("input_tensor", None),
            "input_shape": kwargs.pop("input_shape", None),
            "include_preprocessing": kwargs.pop("include_preprocessing", True),
            "include_top": kwargs.pop("include_top", True),
            "pooling": kwargs.pop("pooling", None),
            "dropout_rate": kwargs.pop("dropout_rate", 0.0),
            "classes": kwargs.pop("classes", 1000),
            "classifier_activation": kwargs.pop(
                "classifier_activation", "softmax"
            ),
            "weights": kwargs.pop("weights", "imagenet"),
            "default_size": kwargs.pop("default_size", default_size),
        }
        return result

    def determine_input_tensor(
        self,
        input_tensor=None,
        input_shape=None,
        default_size=224,
        min_size=32,
        require_flatten=False,
        static_shape=False,
    ):
        """Determine the input tensor by the arguments."""
        input_shape = imagenet_utils.obtain_input_shape(
            input_shape,
            default_size=default_size,
            min_size=min_size,
            data_format="channels_last",  # always channels_last
            require_flatten=require_flatten or static_shape,
            weights=None,
        )

        if input_tensor is None:
            x = layers.Input(shape=input_shape)
        else:
            if not backend.is_keras_tensor(input_tensor):
                x = layers.Input(tensor=input_tensor, shape=input_shape)
            else:
                x = input_tensor
        return x

    def build_preprocessing(self, inputs):
        # TODO: add docstring
        raise NotImplementedError

    def build_top(self, inputs, classes, classifier_activation, dropout_rate):
        # TODO: add docstring
        raise NotImplementedError

    def add_references(self, parsed_kwargs: typing.Dict[str, typing.Any]):
        self.include_preprocessing = parsed_kwargs["include_preprocessing"]
        self.include_top = parsed_kwargs["include_top"]
        self.pooling = parsed_kwargs["pooling"]
        self.dropout_rate = parsed_kwargs["dropout_rate"]
        self.classes = parsed_kwargs["classes"]
        self.classifier_activation = parsed_kwargs["classifier_activation"]
        # `self.weights` is been used internally
        self._weights = parsed_kwargs["weights"]

    @staticmethod
    @abc.abstractmethod
    def available_feature_keys():
        # TODO: add docstring
        raise NotImplementedError

    def get_config(self):
        # Don't chain to super here. The default `get_config()` for functional
        # models is nested and cannot be passed to BaseModel.
        config = {
            "name": self.name,
            "trainable": self.trainable,
            "as_feature_extractor": self.as_feature_extractor,
            "feature_keys": self.feature_keys,
            # common
            "input_shape": self.input_shape[1:],
            "include_preprocessing": self.include_preprocessing,
            "include_top": self.include_top,
            "pooling": self.pooling,
            "dropout_rate": self.dropout_rate,
            "classes": self.classes,
            "classifier_activation": self.classifier_activation,
            "weights": self._weights,
        }
        return config

    def fix_config(self, config: typing.Dict):
        return config

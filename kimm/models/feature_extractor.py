import abc
import typing

from keras import KerasTensor
from keras import models


class FeatureExtractor(models.Model):
    @staticmethod
    @abc.abstractmethod
    def available_feature_keys():
        return []

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

    def get_config(self):
        # Don't chain to super here. The default `get_config()` for functional
        # models is nested and cannot be passed to FeatureExtractor.
        config = {
            "name": self.name,
            "trainable": self.trainable,
            "as_feature_extractor": self.as_feature_extractor,
            "feature_keys": self.feature_keys,
        }
        return config

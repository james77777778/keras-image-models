import typing

# {
#     "name",
#     "support_feature",
#     "available_feature_keys",
#     "has_pretrained",
# }
MODEL_REGISTRY: typing.List[typing.Dict[str, typing.Union[str, bool]]] = []


def clear_registry():
    MODEL_REGISTRY.clear()


def add_model_to_registry(model_cls, has_pretrained=False):
    from kimm.models.feature_extractor import FeatureExtractor

    support_feature = False
    available_feature_keys = []
    if issubclass(model_cls, FeatureExtractor):
        support_feature = True
        available_feature_keys = model_cls.available_feature_keys()
    for info in MODEL_REGISTRY:
        if info["name"] == model_cls.__name__:
            raise ValueError(
                f"MODEL_REGISTRY already contains name={model_cls.__name__}!"
            )
    MODEL_REGISTRY.append(
        {
            "name": model_cls.__name__,
            "support_feature": support_feature,
            "available_feature_keys": available_feature_keys,
            "has_pretrained": has_pretrained,
        }
    )


def list_models(
    name: typing.Optional[str] = None,
    support_feature: typing.Optional[bool] = None,
    has_pretrained: typing.Optional[bool] = None,
):
    result_names: typing.Set = set()
    for info in MODEL_REGISTRY:
        # add by default
        result_names.add(info["name"])
        need_remove = False

        # filter by the args
        if name is not None and name.lower() not in info["name"].lower():
            need_remove = True
        if (
            support_feature is not None
            and info["support_feature"] is not support_feature
        ):
            need_remove = True
        if (
            has_pretrained is not None
            and info["has_pretrained"] is not has_pretrained
        ):
            need_remove = True

        if need_remove:
            result_names.remove(info["name"])
    return sorted(result_names)

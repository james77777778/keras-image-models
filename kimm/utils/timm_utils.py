import typing

import keras
import numpy as np


def _is_useless_weights(name: str):
    if "num_batches_tracked" in name:
        return True
    else:
        return False


def _is_non_trainable_weights(name: str):
    if "running_mean" in name or "running_var" in name:
        return True
    else:
        return False


def separate_torch_state_dict(state_dict: typing.OrderedDict):
    trainable_state_dict = state_dict.copy()
    non_trainable_state_dict = state_dict.copy()
    trainable_remove_keys = []
    non_trainable_remove_keys = []
    for k in state_dict.keys():
        if _is_useless_weights(k):
            trainable_remove_keys.append(k)
            non_trainable_remove_keys.append(k)
            continue
        if _is_non_trainable_weights(k):
            trainable_remove_keys.append(k)
        else:
            non_trainable_remove_keys.append(k)
    for k in trainable_remove_keys:
        trainable_state_dict.pop(k)
    for k in non_trainable_remove_keys:
        non_trainable_state_dict.pop(k)
    return trainable_state_dict, non_trainable_state_dict


def separate_keras_weights(keras_model: keras.Model):
    trainable_weights = []
    non_trainable_weights = []
    for layer in keras_model.layers:
        if hasattr(layer, "extra_layers"):
            for sub_layer in layer.extra_layers:
                sub_layer: keras.Layer
                for weight in sub_layer.trainable_weights:
                    trainable_weights.append(
                        (weight, sub_layer.name + "_" + weight.name)
                    )
                for weight in sub_layer.non_trainable_weights:
                    non_trainable_weights.append(
                        (weight, sub_layer.name + "_" + weight.name)
                    )
        else:
            layer: keras.Layer
            for weight in layer.trainable_weights:
                trainable_weights.append(
                    (weight, layer.name + "_" + weight.name)
                )
            for weight in layer.non_trainable_weights:
                non_trainable_weights.append(
                    (weight, layer.name + "_" + weight.name)
                )
    return trainable_weights, non_trainable_weights


def assign_weights(
    keras_name: str, keras_weight: keras.Variable, torch_weight: np.ndarray
):
    if len(keras_weight.shape) == 4:
        if (
            "conv" in keras_name
            or "pointwise" in keras_name
            or "dwconv2d" in keras_name
            or "depthwise" in keras_name
        ):
            try:
                # conventional conv2d layer
                keras_weight.assign(np.transpose(torch_weight, [2, 3, 1, 0]))
            except ValueError:
                # depthwise conv2d layer
                keras_weight.assign(np.transpose(torch_weight, [2, 3, 0, 1]))
        else:
            raise ValueError(
                f"Failed to assign {keras_name}. "
                f"keras weight shape={keras_weight.shape}, "
                f"torch weight shape={torch_weight.shape}"
            )
    elif len(keras_weight.shape) == 2:
        # dense layer
        keras_weight.assign(np.transpose(torch_weight))
    elif len(keras_weight.shape) == 1:
        keras_weight.assign(torch_weight)
    elif tuple(keras_weight.shape) == tuple(torch_weight.shape):
        keras_weight.assign(torch_weight)
    else:
        raise ValueError(
            f"Failed to assign {keras_name}, "
            f"keras_weight.shape={keras_weight.shape}, "
            f"torch_weight.shape={torch_weight.shape}, "
        )


def is_same_weights(
    keras_name: str,
    keras_weights: keras.Variable,
    torch_name: str,
    torch_weights: np.ndarray,
):
    if np.sum(keras_weights.shape) != np.sum(torch_weights.shape):
        return False
    elif keras_name[-6:] == "kernel" and torch_name[-6:] != "weight":
        # Conv kernel
        return False
    elif keras_name[-5:] == "gamma" and torch_name[-6:] != "weight":
        # BatchNormalization gamma
        return False
    elif keras_name[-4:] == "beta" and torch_name[-4:] != "bias":
        # BatchNormalization beta
        return False
    elif (
        keras_name[-11:] == "moving_mean" and torch_name[-12:] != "running_mean"
    ):
        # BatchNormalization moving_mean
        return False
    elif (
        keras_name[-11:] == "moving_variance"
        and torch_name[-12:] != "running_var"
    ):
        # BatchNormalization moving_variance
        return False
    else:
        # TODO: is it always true?
        return True

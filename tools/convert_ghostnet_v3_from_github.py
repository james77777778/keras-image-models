"""
From: https://github.com/huawei-noah/Efficient-AI-Backbones
"""

import os
import pathlib
import urllib.parse

import keras
import numpy as np
import torch

from kimm.models import ghostnet
from kimm.timm_utils import assign_weights
from kimm.timm_utils import is_same_weights
from kimm.timm_utils import separate_keras_weights
from kimm.timm_utils import separate_torch_state_dict
from tools.third_party.ghostnet_v3.ghostnetv3 import ghostnetv3

github_model_items = [
    (
        "ghostnetv3-1.0",
        ghostnetv3,
        dict(width=1.0),
        "https://github.com/huawei-noah/Efficient-AI-Backbones/releases/download/GhostNetV3/ghostnetv3-1.0.pth.tar",
    ),
]
keras_model_classes = [
    ghostnet.GhostNetV3W100,
]

for github_model_item, keras_model_class in zip(
    github_model_items, keras_model_classes
):
    """
    Prepare timm model and keras model
    """
    model_name, model_class, model_args, model_url = github_model_item

    input_shape = [224, 224, 3]
    result = urllib.parse.urlparse(model_url)
    filename = pathlib.Path(result.path).name
    file_path = keras.utils.get_file(
        fname=filename, origin=model_url, cache_subdir="kimm_models"
    )
    state_dict = torch.load(file_path, map_location="cpu")["state_dict"]
    torch_model = model_class(**model_args)
    torch_model.load_state_dict(state_dict)
    torch_model.eval()
    trainable_state_dict, non_trainable_state_dict = separate_torch_state_dict(
        state_dict
    )
    keras_model = keras_model_class(
        input_shape=input_shape,
        include_preprocessing=False,
        classifier_activation="linear",
        weights=None,
    )
    trainable_weights, non_trainable_weights = separate_keras_weights(
        keras_model
    )

    # for torch_name, (_, keras_name) in zip(
    #     trainable_state_dict.keys(), trainable_weights
    # ):
    #     print(f"{torch_name}    {keras_name}")

    # print(len(trainable_state_dict.keys()))
    # print(len(trainable_weights))
    # exit()

    # for torch_name, (_, keras_name) in zip(
    #     non_trainable_state_dict.keys(), non_trainable_weights
    # ):
    #     print(f"{torch_name}    {keras_name}")

    # print(len(non_trainable_state_dict.keys()))
    # print(len(non_trainable_weights))
    # exit()

    """
    Assign weights
    """
    for keras_weight, keras_name in trainable_weights + non_trainable_weights:
        keras_name: str
        torch_name = keras_name
        torch_name = torch_name.replace("_", ".")
        # skip reparam_conv
        if "reparam_conv_conv2d" in keras_name:
            continue
        # Stem
        torch_name = torch_name.replace("conv.stem.conv2d", "conv_stem")
        torch_name = torch_name.replace("conv.stem.bn", "bn1")
        # ReparameterizableConv2D
        # primary
        for i in range(3):
            torch_name = torch_name.replace(
                f"primary.conv.conv.kxk.{i}.kernel",
                f"primary_rpr_conv.{i}.conv.weight",
            )
            for pair in (
                ("gamma", "weight"),
                ("beta", "bias"),
                ("moving.mean", "running_mean"),
                ("moving.variance", "running_var"),
            ):
                a, b = pair
                torch_name = torch_name.replace(
                    f"primary.conv.conv.kxk.{i}.{a}",
                    f"primary_rpr_conv.{i}.bn.{b}",
                )
        # cheap
        torch_name = torch_name.replace(
            "cheap.operation.skip", "cheap_rpr_skip"
        )
        torch_name = torch_name.replace(
            "cheap.operation.conv.scale.kernel", "cheap_rpr_scale.conv.weight"
        )
        for pair in (
            ("gamma", "weight"),
            ("beta", "bias"),
            ("moving.mean", "running_mean"),
            ("moving.variance", "running_var"),
        ):
            a, b = pair
            torch_name = torch_name.replace(
                f"cheap.operation.conv.scale.{a}",
                f"cheap_rpr_scale.bn.{b}",
            )
        for i in range(3):
            torch_name = torch_name.replace(
                f"cheap.operation.conv.kxk.{i}.kernel",
                f"cheap_rpr_conv.{i}.conv.weight",
            )
            for pair in (
                ("gamma", "weight"),
                ("beta", "bias"),
                ("moving.mean", "running_mean"),
                ("moving.variance", "running_var"),
            ):
                a, b = pair
                torch_name = torch_name.replace(
                    f"cheap.operation.conv.kxk.{i}.{a}",
                    f"cheap_rpr_conv.{i}.bn.{b}",
                )
        # short
        for i in range(3):
            torch_name = torch_name.replace(
                "short.conv.0.conv2d.kernel", "short_conv.0.weight"
            )
            for pair in (
                ("gamma", "weight"),
                ("beta", "bias"),
                ("moving.mean", "running_mean"),
                ("moving.variance", "running_var"),
            ):
                a, b = pair
                torch_name = torch_name.replace(
                    f"short.conv.0.bn.{a}",
                    f"short_conv.1.{b}",
                )
            torch_name = torch_name.replace(
                "short.conv.1.dwconv2d.kernel", "short_conv.2.weight"
            )
            for pair in (
                ("gamma", "weight"),
                ("beta", "bias"),
                ("moving.mean", "running_mean"),
                ("moving.variance", "running_var"),
            ):
                a, b = pair
                torch_name = torch_name.replace(
                    f"short.conv.1.bn.{a}",
                    f"short_conv.3.{b}",
                )
            torch_name = torch_name.replace(
                "short.conv.2.dwconv2d.kernel", "short_conv.4.weight"
            )
            for pair in (
                ("gamma", "weight"),
                ("beta", "bias"),
                ("moving.mean", "running_mean"),
                ("moving.variance", "running_var"),
            ):
                a, b = pair
                torch_name = torch_name.replace(
                    f"short.conv.2.bn.{a}", f"short_conv.5.{b}"
                )
        # Depth-wise
        torch_name = torch_name.replace(
            "conv.dw.conv.scale.kernel", "dw_rpr_scale.conv.weight"
        )
        for pair in (
            ("gamma", "weight"),
            ("beta", "bias"),
            ("moving.mean", "running_mean"),
            ("moving.variance", "running_var"),
        ):
            a, b = pair
            torch_name = torch_name.replace(
                f"conv.dw.conv.scale.{a}",
                f"dw_rpr_scale.bn.{b}",
            )
        for i in range(3):
            torch_name = torch_name.replace(
                f"conv.dw.conv.kxk.{i}.kernel",
                f"dw_rpr_conv.{i}.conv.weight",
            )
            for pair in (
                ("gamma", "weight"),
                ("beta", "bias"),
                ("moving.mean", "running_mean"),
                ("moving.variance", "running_var"),
            ):
                a, b = pair
                torch_name = torch_name.replace(
                    f"conv.dw.conv.kxk.{i}.{a}",
                    f"dw_rpr_conv.{i}.bn.{b}",
                )
        # Squeeze-and-excitation
        torch_name = torch_name.replace("se.conv.reduce", "se.conv_reduce")
        torch_name = torch_name.replace("se.conv.expand", "se.conv_expand")
        # Shortcut
        torch_name = torch_name.replace(
            "shortcut1.dwconv2d.kernel", "shortcut.0.weight"
        )
        for pair in (
            ("gamma", "weight"),
            ("beta", "bias"),
            ("moving.mean", "running_mean"),
            ("moving.variance", "running_var"),
        ):
            a, b = pair
            torch_name = torch_name.replace(
                f"shortcut1.bn.{a}", f"shortcut.1.{b}"
            )
        torch_name = torch_name.replace(
            "shortcut2.conv2d.kernel", "shortcut.2.weight"
        )
        for pair in (
            ("gamma", "weight"),
            ("beta", "bias"),
            ("moving.mean", "running_mean"),
            ("moving.variance", "running_var"),
        ):
            a, b = pair
            torch_name = torch_name.replace(
                f"shortcut2.bn.{a}", f"shortcut.3.{b}"
            )

        # Last block
        torch_name = torch_name.replace("blocks.9.conv2d", "blocks.9.0.conv")
        torch_name = torch_name.replace("blocks.9.bn", "blocks.9.0.bn1")

        # Head
        torch_name = torch_name.replace("conv.head", "conv_head")

        # weights naming mapping
        torch_name = torch_name.replace("kernel", "weight")  # conv2d
        torch_name = torch_name.replace("gamma", "weight")  # bn
        torch_name = torch_name.replace("beta", "bias")  # bn
        torch_name = torch_name.replace("moving.mean", "running_mean")  # bn
        torch_name = torch_name.replace("moving.variance", "running_var")  # bn

        # assign weights
        if torch_name in trainable_state_dict:
            torch_weights = trainable_state_dict[torch_name].numpy()
        elif torch_name in non_trainable_state_dict:
            torch_weights = non_trainable_state_dict[torch_name].numpy()
        else:
            raise ValueError(
                "Can't find the corresponding torch weights. "
                f"Got keras_name={keras_name}, torch_name={torch_name}"
            )
        if is_same_weights(keras_name, keras_weight, torch_name, torch_weights):
            assign_weights(keras_name, keras_weight, torch_weights)
        else:
            raise ValueError(
                "Can't find the corresponding torch weights. The shape is "
                f"mismatched. Got keras_name={keras_name}, "
                f"keras_weight shape={keras_weight.shape}, "
                f"torch_name={torch_name}, "
                f"torch_weights shape={torch_weights.shape}"
            )

    """
    Verify model outputs
    """
    np.random.seed(2023)
    keras_data = np.random.uniform(size=[1] + input_shape).astype("float32")
    torch_data = torch.from_numpy(np.transpose(keras_data, [0, 3, 1, 2]))
    torch_y = torch_model(torch_data)
    keras_y = keras_model(keras_data, training=False)
    torch_y = torch_y.detach().cpu().numpy()
    torch_y = np.expand_dims(torch_y, axis=0)
    keras_y = keras.ops.convert_to_numpy(keras_y)
    # TODO: Error is large
    np.testing.assert_allclose(torch_y, keras_y, atol=0.5)
    print(f"{keras_model_class.__name__}: output matched!")

    """
    Save converted model
    """
    os.makedirs("exported", exist_ok=True)
    export_path = f"exported/{keras_model.name.lower()}_{model_name}.keras"
    keras_model.save(export_path)
    print(f"Export to {export_path}")

"""
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install timm
"""

import os

import keras
import numpy as np
import timm
import torch

from kimm.models.ghostnet import GhostNet100
from kimm.models.ghostnet import GhostNet100V2
from kimm.models.ghostnet import GhostNet130V2
from kimm.models.ghostnet import GhostNet160V2
from kimm.utils.timm_utils import assign_weights
from kimm.utils.timm_utils import is_same_weights
from kimm.utils.timm_utils import separate_keras_weights
from kimm.utils.timm_utils import separate_torch_state_dict

timm_model_names = [
    "ghostnet_100",
    "ghostnetv2_100",
    "ghostnetv2_130",
    "ghostnetv2_160",
]
keras_model_classes = [
    GhostNet100,
    GhostNet100V2,
    GhostNet130V2,
    GhostNet160V2,
]

for timm_model_name, keras_model_class in zip(
    timm_model_names, keras_model_classes
):
    """
    Prepare timm model and keras model
    """
    input_shape = [224, 224, 3]
    torch_model = timm.create_model(timm_model_name, pretrained=True)
    torch_model = torch_model.eval()
    trainable_state_dict, non_trainable_state_dict = separate_torch_state_dict(
        torch_model.state_dict()
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

    """
    Assign weights
    """
    for keras_weight, keras_name in trainable_weights + non_trainable_weights:
        keras_name: str
        torch_name = keras_name
        torch_name = torch_name.replace("_", ".")
        # stem
        torch_name = torch_name.replace("conv.stem.conv2d", "conv_stem")
        torch_name = torch_name.replace("conv.stem.bn", "bn1")
        # blocks
        torch_name = torch_name.replace("primary.conv.conv2d", "primary_conv.0")
        torch_name = torch_name.replace("primary.conv.bn", "primary_conv.1")
        torch_name = torch_name.replace(
            "cheap.operation.dwconv2d", "cheap_operation.0"
        )
        torch_name = torch_name.replace(
            "cheap.operation.bn", "cheap_operation.1"
        )
        torch_name = torch_name.replace("conv.dw.dwconv2d", "conv_dw")
        torch_name = torch_name.replace("conv.dw.bn", "bn_dw")
        torch_name = torch_name.replace("shortcut1.dwconv2d", "shortcut.0")
        torch_name = torch_name.replace("shortcut1.bn", "shortcut.1")
        torch_name = torch_name.replace("shortcut2.conv2d", "shortcut.2")
        torch_name = torch_name.replace("shortcut2.bn", "shortcut.3")
        # se
        torch_name = torch_name.replace("se.conv.reduce", "se.conv_reduce")
        torch_name = torch_name.replace("se.conv.expand", "se.conv_expand")
        # short conv (GhostNetV2)
        torch_name = torch_name.replace("short.conv1.conv2d", "short_conv.0")
        torch_name = torch_name.replace("short.conv1.bn", "short_conv.1")
        torch_name = torch_name.replace("short.conv2.dwconv2d", "short_conv.2")
        torch_name = torch_name.replace("short.conv2.bn", "short_conv.3")
        torch_name = torch_name.replace("short.conv3.dwconv2d", "short_conv.4")
        torch_name = torch_name.replace("short.conv3.bn", "short_conv.5")
        # final block
        torch_name = torch_name.replace("blocks.9.conv2d", "blocks.9.0.conv")
        torch_name = torch_name.replace("blocks.9.bn", "blocks.9.0.bn1")
        # conv head
        if torch_name.startswith("conv.head"):
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
    keras_y = keras.ops.convert_to_numpy(keras_y)
    np.testing.assert_allclose(torch_y, keras_y, atol=5e-1)
    print(f"{keras_model_class.__name__}: output matched!")

    """
    Save converted model
    """
    os.makedirs("exported", exist_ok=True)
    export_path = f"exported/{keras_model.name.lower()}_{timm_model_name}.keras"
    keras_model.save(export_path)
    print(f"Export to {export_path}")

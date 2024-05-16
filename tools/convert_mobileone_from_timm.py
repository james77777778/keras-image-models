"""
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install timm
"""

import os

import keras
import numpy as np
import timm
import torch

from kimm.models import mobileone
from kimm.timm_utils import assign_weights
from kimm.timm_utils import is_same_weights
from kimm.timm_utils import separate_keras_weights
from kimm.timm_utils import separate_torch_state_dict

timm_model_names = [
    "mobileone_s0.apple_in1k",
    "mobileone_s1.apple_in1k",
    "mobileone_s2.apple_in1k",
    "mobileone_s3.apple_in1k",
    # "mobileone_s4.apple_in1k",
]
keras_model_classes = [
    mobileone.MobileOneS0,
    mobileone.MobileOneS1,
    mobileone.MobileOneS2,
    mobileone.MobileOneS3,
    # mobileone.MobileOneS4,
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
        # mobile_one_conv2d
        if "conv.kxk" in torch_name and "kernel" in torch_name:
            torch_name = torch_name.replace("conv.kxk", "conv_kxk")
            torch_name = torch_name.replace("kernel", "conv.kernel")
        if "conv.kxk" in torch_name and "gamma" in torch_name:
            torch_name = torch_name.replace("conv.kxk", "conv_kxk")
            torch_name = torch_name.replace("gamma", "bn.gamma")
        if "conv.kxk" in torch_name and "beta" in torch_name:
            torch_name = torch_name.replace("conv.kxk", "conv_kxk")
            torch_name = torch_name.replace("beta", "bn.beta")
        torch_name = torch_name.replace(
            "conv.scale.kernel", "conv_scale.conv.kernel"
        )
        torch_name = torch_name.replace(
            "conv.scale.gamma", "conv_scale.bn.gamma"
        )
        torch_name = torch_name.replace("conv.scale.beta", "conv_scale.bn.beta")
        # mobile_one_conv2d bn
        if "conv.kxk" in torch_name and "moving.mean" in torch_name:
            torch_name = torch_name.replace("conv.kxk", "conv_kxk")
            torch_name = torch_name.replace("moving.mean", "bn.moving.mean")
        if "conv.kxk" in torch_name and "moving.variance" in torch_name:
            torch_name = torch_name.replace("conv.kxk", "conv_kxk")
            torch_name = torch_name.replace(
                "moving.variance", "bn.moving.variance"
            )
        torch_name = torch_name.replace(
            "conv.scale.moving.mean", "conv_scale.bn.moving.mean"
        )
        torch_name = torch_name.replace(
            "conv.scale.moving.variance", "conv_scale.bn.moving.variance"
        )
        # head
        torch_name = torch_name.replace("classifier", "head.fc")

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
    np.testing.assert_allclose(torch_y, keras_y, atol=1e-4)
    print(f"{keras_model_class.__name__}: output matched!")

    """
    Save converted model
    """
    os.makedirs("exported", exist_ok=True)
    export_path = f"exported/{keras_model.name.lower()}_{timm_model_name}.keras"
    keras_model.save(export_path)
    print(f"Export to {export_path}")

"""
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install timm
"""

import os

import keras
import numpy as np
import timm
import torch

from kimm.models import efficientnet
from kimm.utils.timm_utils import assign_weights
from kimm.utils.timm_utils import is_same_weights
from kimm.utils.timm_utils import separate_keras_weights
from kimm.utils.timm_utils import separate_torch_state_dict

timm_model_names = [
    "tf_efficientnet_b0.ns_jft_in1k",
    "tf_efficientnet_b1.ns_jft_in1k",
    "tf_efficientnet_b2.ns_jft_in1k",
    "tf_efficientnet_b3.ns_jft_in1k",
    "tf_efficientnet_b4.ns_jft_in1k",
    "tf_efficientnet_b5.ns_jft_in1k",
    "tf_efficientnet_b6.ns_jft_in1k",
    "tf_efficientnet_b7.ns_jft_in1k",
    "tf_efficientnet_lite0.in1k",
    "tf_efficientnet_lite1.in1k",
    "tf_efficientnet_lite2.in1k",
    "tf_efficientnet_lite3.in1k",
    "tf_efficientnet_lite4.in1k",
    "tf_efficientnetv2_s.in21k_ft_in1k",
    "tf_efficientnetv2_m.in21k_ft_in1k",
    "tf_efficientnetv2_l.in21k_ft_in1k",
    "tf_efficientnetv2_xl.in21k_ft_in1k",
    "tf_efficientnetv2_b0.in1k",
    "tf_efficientnetv2_b1.in1k",
    "tf_efficientnetv2_b2.in1k",
    "tf_efficientnetv2_b3.in1k",
    "tinynet_a.in1k",
    "tinynet_b.in1k",
    "tinynet_c.in1k",
    "tinynet_d.in1k",
    "tinynet_e.in1k",
]
keras_model_classes = [
    efficientnet.EfficientNetB0,
    efficientnet.EfficientNetB1,
    efficientnet.EfficientNetB2,
    efficientnet.EfficientNetB3,
    efficientnet.EfficientNetB4,
    efficientnet.EfficientNetB5,
    efficientnet.EfficientNetB6,
    efficientnet.EfficientNetB7,
    efficientnet.EfficientNetLiteB0,
    efficientnet.EfficientNetLiteB1,
    efficientnet.EfficientNetLiteB2,
    efficientnet.EfficientNetLiteB3,
    efficientnet.EfficientNetLiteB4,
    efficientnet.EfficientNetV2S,
    efficientnet.EfficientNetV2M,
    efficientnet.EfficientNetV2L,
    efficientnet.EfficientNetV2XL,
    efficientnet.EfficientNetV2B0,
    efficientnet.EfficientNetV2B1,
    efficientnet.EfficientNetV2B2,
    efficientnet.EfficientNetV2B3,
    efficientnet.TinyNetA,
    efficientnet.TinyNetB,
    efficientnet.TinyNetC,
    efficientnet.TinyNetD,
    efficientnet.TinyNetE,
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
        if "EfficientNetV2" in keras_model_class.__name__:
            if "blocks.0" in torch_name:
                # normal conv
                torch_name = torch_name.replace("conv2d", "conv")
                torch_name = torch_name.replace("bn", "bn1")
            elif "blocks.1" in torch_name or "blocks.2" in torch_name:
                # edge residual block
                torch_name = torch_name.replace("conv.exp.conv2d", "conv_exp")
                torch_name = torch_name.replace("conv.exp.bn", "bn1")
                torch_name = torch_name.replace("conv.pwl.conv2d", "conv_pwl")
                torch_name = torch_name.replace("conv.pwl.bn", "bn2")
        else:
            if "blocks.0" in torch_name:
                # depthwise separation block
                torch_name = torch_name.replace("conv.dw.dwconv2d", "conv_dw")
                torch_name = torch_name.replace("conv.dw.bn", "bn1")
                torch_name = torch_name.replace("conv.pw.conv2d", "conv_pw")
                torch_name = torch_name.replace("conv.pw.bn", "bn2")
        # inverted residual block
        torch_name = torch_name.replace("conv.pw.conv2d", "conv_pw")
        torch_name = torch_name.replace("conv.pw.bn", "bn1")
        torch_name = torch_name.replace("conv.dw.dwconv2d", "conv_dw")
        torch_name = torch_name.replace("conv.dw.bn", "bn2")
        torch_name = torch_name.replace("conv.pwl.conv2d", "conv_pwl")
        torch_name = torch_name.replace("conv.pwl.bn", "bn3")
        # se
        torch_name = torch_name.replace("se.conv.reduce", "se.conv_reduce")
        torch_name = torch_name.replace("se.conv.expand", "se.conv_expand")
        # conv head
        torch_name = torch_name.replace("conv.head.conv2d", "conv_head")
        torch_name = torch_name.replace("conv.head.bn", "bn2")

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
    np.testing.assert_allclose(torch_y, keras_y, atol=2e-5)
    print(f"{keras_model_class.__name__}: output matched!")

    """
    Save converted model
    """
    os.makedirs("exported", exist_ok=True)
    export_path = f"exported/{keras_model.name.lower()}_{timm_model_name}.keras"
    keras_model.save(export_path)
    print(f"Export to {export_path}")

"""
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install timm
"""

import os

import keras
import numpy as np
import timm
import torch

from kimm.models import mobilevit
from kimm.timm_utils import assign_weights
from kimm.timm_utils import is_same_weights
from kimm.timm_utils import separate_keras_weights
from kimm.timm_utils import separate_torch_state_dict

timm_model_names = [
    "mobilevit_xxs.cvnets_in1k",
    "mobilevit_xs.cvnets_in1k",
    "mobilevit_s.cvnets_in1k",
    "mobilevitv2_050.cvnets_in1k",
    "mobilevitv2_075.cvnets_in1k",
    "mobilevitv2_100.cvnets_in1k",
    "mobilevitv2_125.cvnets_in1k",
    "mobilevitv2_150.cvnets_in22k_ft_in1k_384",
    "mobilevitv2_175.cvnets_in22k_ft_in1k_384",
    "mobilevitv2_200.cvnets_in22k_ft_in1k_384",
]
keras_model_classes = [
    mobilevit.MobileViTXXS,
    mobilevit.MobileViTXS,
    mobilevit.MobileViTS,
    mobilevit.MobileViTV2W050,
    mobilevit.MobileViTV2W075,
    mobilevit.MobileViTV2W100,
    mobilevit.MobileViTV2W125,
    mobilevit.MobileViTV2W150,
    mobilevit.MobileViTV2W175,
    mobilevit.MobileViTV2W200,
]

for timm_model_name, keras_model_class in zip(
    timm_model_names, keras_model_classes
):
    """
    Prepare timm model and keras model
    """
    input_shape = [256, 256, 3]  # use size of 384 for best performance
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
        torch_name = torch_name.replace("stem.conv2d", "stem.conv")
        # inverted residual block
        torch_name = torch_name.replace("conv.pw.conv2d", "conv1_1x1.conv")
        torch_name = torch_name.replace("conv.pw.bn", "conv1_1x1.bn")
        torch_name = torch_name.replace("conv.dw.dwconv2d", "conv2_kxk.conv")
        torch_name = torch_name.replace("conv.dw.bn", "conv2_kxk.bn")
        torch_name = torch_name.replace("conv.pwl.conv2d", "conv3_1x1.conv")
        torch_name = torch_name.replace("conv.pwl.bn", "conv3_1x1.bn")
        # mobilevit block
        torch_name = torch_name.replace("conv.kxk.conv2d", "conv_kxk.conv")
        torch_name = torch_name.replace("conv.kxk.bn", "conv_kxk.bn")
        torch_name = torch_name.replace("conv.1x1", "conv_1x1")
        torch_name = torch_name.replace("attn", "attn.qkv")
        # torch_name = torch_name.replace("attn", "attn.proj")
        torch_name = torch_name.replace("conv.proj.conv2d", "conv_proj.conv")
        torch_name = torch_name.replace("conv.proj.bn", "conv_proj.bn")
        torch_name = torch_name.replace(
            "conv.fusion.conv2d", "conv_fusion.conv"
        )
        torch_name = torch_name.replace("conv.fusion.bn", "conv_fusion.bn")
        # mobilevitv2 block
        torch_name = torch_name.replace("conv.kxk.dwconv2d", "conv_kxk.conv")
        torch_name = torch_name.replace(
            "attn.qkv.qkv.proj.conv2d", "attn.qkv_proj"
        )
        torch_name = torch_name.replace(
            "attn.qkv.out.proj.conv2d", "attn.out_proj"
        )
        torch_name = torch_name.replace("mlp.fc1.conv2d", "mlp.fc1")
        torch_name = torch_name.replace("mlp.fc2.conv2d", "mlp.fc2")
        # final block
        torch_name = torch_name.replace("final.conv.conv2d", "final_conv.conv")
        torch_name = torch_name.replace("final.conv.bn", "final_conv.bn")
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
        # special case for Attention module
        elif "attn" in keras_name:
            torch_name = torch_name.replace("attn.qkv", "attn.proj")
            torch_weights = trainable_state_dict[torch_name].numpy()
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
    np.testing.assert_allclose(torch_y, keras_y, atol=1e-3)
    print(f"{keras_model_class.__name__}: output matched!")

    """
    Save converted model
    """
    os.makedirs("exported", exist_ok=True)
    export_path = f"exported/{keras_model.name.lower()}_{timm_model_name}.keras"
    keras_model.save(export_path)
    print(f"Export to {export_path}")

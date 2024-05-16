"""
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install timm
"""

import os

import keras
import numpy as np
import timm
import torch

from kimm.models import convnext
from kimm.timm_utils import assign_weights
from kimm.timm_utils import is_same_weights
from kimm.timm_utils import separate_keras_weights
from kimm.timm_utils import separate_torch_state_dict

timm_model_names = [
    "convnext_atto.d2_in1k",
    "convnext_femto.d1_in1k",
    "convnext_pico.d1_in1k",
    "convnext_nano.in12k_ft_in1k",
    "convnext_tiny.in12k_ft_in1k",
    "convnext_small.in12k_ft_in1k",
    "convnext_base.fb_in22k_ft_in1k",
    "convnext_large.fb_in22k_ft_in1k",
    "convnext_xlarge.fb_in22k_ft_in1k",
]
keras_model_classes = [
    convnext.ConvNeXtAtto,
    convnext.ConvNeXtFemto,
    convnext.ConvNeXtPico,
    convnext.ConvNeXtNano,
    convnext.ConvNeXtTiny,
    convnext.ConvNeXtSmall,
    convnext.ConvNeXtBase,
    convnext.ConvNeXtLarge,
    convnext.ConvNeXtXLarge,
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
        # prevent gamma to be replaced
        is_layerscale = False
        keras_name: str
        torch_name = keras_name
        torch_name = torch_name.replace("_", ".")

        # stem
        torch_name = torch_name.replace("stem.0.conv2d.kernel", "stem.0.weight")
        torch_name = torch_name.replace("stem.0.conv2d.bias", "stem.0.bias")

        # blocks
        torch_name = torch_name.replace("dwconv2d.", "")
        torch_name = torch_name.replace("conv2d.", "")
        torch_name = torch_name.replace("conv.dw", "conv_dw")
        if "layerscale" in torch_name:
            is_layerscale = True
        torch_name = torch_name.replace("layerscale.", "")
        # head
        torch_name = torch_name.replace("classifier", "head.fc")

        # weights naming mapping
        torch_name = torch_name.replace("kernel", "weight")  # conv2d
        if not is_layerscale:
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
        if is_layerscale:
            assign_weights(keras_name, keras_weight, torch_weights)
        elif is_same_weights(
            keras_name, keras_weight, torch_name, torch_weights
        ):
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

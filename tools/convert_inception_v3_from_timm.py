"""
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install timm
"""
import os

import keras
import numpy as np
import timm
import torch

from kimm.models import inception_v3
from kimm.utils.timm_utils import assign_weights
from kimm.utils.timm_utils import is_same_weights
from kimm.utils.timm_utils import separate_keras_weights
from kimm.utils.timm_utils import separate_torch_state_dict

timm_model_names = [
    "inception_v3.gluon_in1k",
]
keras_model_classes = [
    inception_v3.InceptionV3,
]

for timm_model_name, keras_model_class in zip(
    timm_model_names, keras_model_classes
):
    """
    Prepare timm model and keras model
    """
    input_shape = [299, 299, 3]
    torch_model = timm.create_model(
        timm_model_name, pretrained=True, aux_logits=False
    )
    torch_model = torch_model.eval()
    trainable_state_dict, non_trainable_state_dict = separate_torch_state_dict(
        torch_model.state_dict()
    )
    keras_model = keras_model_class(
        has_aux_logits=False,
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
    Preprocess
    """
    new_dict = {}
    old_keys = trainable_state_dict.keys()
    new_keys = []
    for k in old_keys:
        new_key = k.replace("_", ".")
        new_key = new_key.replace("running.mean", "running_mean")
        new_key = new_key.replace("running.var", "running_var")
        new_keys.append(new_key)
    for k1, k2 in zip(trainable_state_dict.keys(), new_keys):
        new_dict[k2] = trainable_state_dict[k1]
    trainable_state_dict = new_dict

    new_dict = {}
    old_keys = non_trainable_state_dict.keys()
    new_keys = []
    for k in old_keys:
        new_key = k.replace("_", ".")
        new_key = new_key.replace("running.mean", "running_mean")
        new_key = new_key.replace("running.var", "running_var")
        new_keys.append(new_key)
    for k1, k2 in zip(non_trainable_state_dict.keys(), new_keys):
        new_dict[k2] = non_trainable_state_dict[k1]
    non_trainable_state_dict = new_dict

    """
    Assign weights
    """
    for keras_weight, keras_name in trainable_weights + non_trainable_weights:
        keras_name: str
        torch_name = keras_name
        torch_name = torch_name.replace("_", ".")
        # general
        torch_name = torch_name.replace("conv2d", "conv")
        # head
        torch_name = torch_name.replace("classifier", "fc")

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
    np.testing.assert_allclose(torch_y, keras_y, atol=1e-5)
    print(f"{keras_model_class.__name__}: output matched!")

    """
    Save converted model
    """
    os.makedirs("exported", exist_ok=True)
    export_path = f"exported/{keras_model.name.lower()}_{timm_model_name}.keras"
    keras_model.save(export_path)
    print(f"Export to {export_path}")

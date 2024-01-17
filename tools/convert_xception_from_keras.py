"""
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install timm
"""
import os
import tempfile

import keras
import numpy as np

from kimm.models import xception

ori_model_classes = [
    keras.applications.Xception,
]
keras_model_classes = [
    xception.Xception,
]

for ori_model_class, keras_model_class in zip(
    ori_model_classes, keras_model_classes
):
    """
    Prepare timm model and keras model
    """
    input_shape = (299, 299, 3)
    ori_model = ori_model_class(
        input_shape=input_shape, classifier_activation="linear"
    )
    keras_model = keras_model_class(
        input_shape=input_shape,
        include_preprocessing=False,
        classifier_activation="linear",
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        ori_model.save_weights(temp_dir + "/model.weights.h5")
        keras_model.load_weights(temp_dir + "/model.weights.h5")

    """
    Verify model outputs
    """
    np.random.seed(2023)
    keras_data = np.random.uniform(size=[1] + list(input_shape)).astype(
        "float32"
    )
    ori_y = ori_model(keras_data, training=False)
    keras_y = keras_model(keras_data, training=False)
    ori_y = keras.ops.convert_to_numpy(ori_y)
    keras_y = keras.ops.convert_to_numpy(keras_y)
    np.testing.assert_allclose(ori_y, keras_y, atol=1e-5)
    print(f"{keras_model_class.__name__}: output matched!")

    """
    Save converted model
    """
    os.makedirs("exported", exist_ok=True)
    export_path = f"exported/{keras_model.name.lower()}.keras"
    keras_model.save(export_path)
    print(f"Export to {export_path}")

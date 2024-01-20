<!-- markdownlint-disable MD033 -->
<!-- markdownlint-disable MD041 -->

# Keras Image Models

<div align="center">
<img width="50%" src="https://github.com/james77777778/kimm/assets/20734616/b21db8f2-307b-4791-b93d-e913e45fb238" alt="KIMM">

[![PyPI](https://img.shields.io/pypi/v/kimm)](https://pypi.org/project/kimm/)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/james77777778/kimm/issues)
[![codecov](https://codecov.io/gh/james77777778/kimm/graph/badge.svg?token=eEha1SR80D)](https://codecov.io/gh/james77777778/kimm)
</div>

## Description

**K**eras **Im**age **M**odels (`kimm`) is a collection of image models, blocks and layers written in Keras 3. The goal is to offer SOTA models with pretrained weights in a user-friendly manner.

## Installation

```bash
# In a working [jax/tensorflow/torch/numpy] backend environment
pip install keras kimm
```

## Quickstart

### Image Classification Using the Model Pretrained on ImageNet

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14WxYgVjlwCIO9MwqPYW-dskbTL2UHsVN?usp=sharing)

```python
import cv2
import keras
from keras import ops
from keras.applications.imagenet_utils import decode_predictions

import kimm

# Use `kimm.list_models` to get the list of available models
print(kimm.list_models())

# Specify the name and other arguments to filter the result
print(kimm.list_models("efficientnet", weights="imagenet"))  # fuzzy search

# Initialize the model with pretrained weights
model = kimm.models.EfficientNetV2B0()
image_size = model._default_size

# Load an image as the model input
image_path = keras.utils.get_file(
    "african_elephant.jpg", "https://i.imgur.com/Bvro0YD.png"
)
image = cv2.imread(image_path)
image = cv2.resize(image, (image_size, image_size))
x = ops.convert_to_tensor(image)
x = ops.expand_dims(x, axis=0)

# Predict
preds = model.predict(x)
print("Predicted:", decode_predictions(preds, top=3)[0])
```

```bash
['ConvMixer1024D20', 'ConvMixer1536D20', 'ConvMixer736D32', 'ConvNeXtAtto', ...]
['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', ...]
1/1 ━━━━━━━━━━━━━━━━━━━━ 11s 11s/step
Predicted: [('n02504458', 'African_elephant', 0.90578836), ('n01871265', 'tusker', 0.024864597), ('n02504013', 'Indian_elephant', 0.01161992)]
```

### An end-to-end example: fine-tuning an image classification model on a cats vs. dogs dataset

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IbqfqG2NKEOKvBOznIPT1kjOdVPfThmd?usp=sharing)

Using `kimm.models.EfficientNetLiteB0`:

<div align="center">
<img width="75%" src="https://github.com/james77777778/kimm/assets/20734616/cbfc0773-a3fa-407d-be9a-fba4f19da6d3" alt="kimm_prediction_0">

<img width="75%" src="https://github.com/james77777778/kimm/assets/20734616/2eac0831-75bb-4790-a3af-412c3e09cf8f" alt="kimm_prediction_1">
</div>

Reference: [Transfer learning & fine-tuning (keras.io)](https://keras.io/guides/transfer_learning/#an-endtoend-example-finetuning-an-image-classification-model-on-a-cats-vs-dogs-dataset)

### Grad-CAM

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1h25VmsYDOLL6BNbRPEVOh1arIgcEoHu6?usp=sharing)

Using `kimm.models.MobileViTS`:

<img width="75%" src="https://github.com/james77777778/kimm/assets/20734616/2e61ae06-32e2-4495-ac3d-04e908dea378" alt="grad_cam">
</div>

Reference: [Grad-CAM class activation visualization (keras.io)](https://keras.io/examples/vision/grad_cam/)

## Model Zoo

|Model|Paper|Weights are ported from|API|
|-|-|-|-|
|ConvMixer|[ICLR 2022 Submission](https://arxiv.org/abs/2201.09792)|`timm`|`kimm.models.ConvMixer*`|
|ConvNeXt|[CVPR 2022](https://arxiv.org/abs/2201.03545)|`timm`|`kimm.models.ConvNeXt*`|
|DenseNet|[CVPR 2017](https://arxiv.org/abs/1608.06993)|`timm`|`kimm.models.DenseNet*`|
|EfficientNet|[ICML 2019](https://arxiv.org/abs/1905.11946)|`timm`|`kimm.models.EfficientNet*`|
|EfficientNetLite|[ICML 2019](https://arxiv.org/abs/1905.11946)|`timm`|`kimm.models.EfficientNetLite*`|
|EfficientNetV2|[ICML 2021](https://arxiv.org/abs/2104.00298)|`timm`|`kimm.models.EfficientNetV2*`|
|GhostNet|[CVPR 2020](https://arxiv.org/abs/1911.11907)|`timm`|`kimm.models.GhostNet*`|
|GhostNetV2|[NeurIPS 2022](https://arxiv.org/abs/2211.12905)|`timm`|`kimm.models.GhostNetV2*`|
|InceptionV3|[CVPR 2016](https://arxiv.org/abs/1512.00567)|`timm`|`kimm.models.InceptionV3`|
|LCNet|[arXiv 2021](https://arxiv.org/abs/2109.15099)|`timm`|`kimm.models.LCNet*`|
|MobileNetV2|[CVPR 2018](https://arxiv.org/abs/1801.04381)|`timm`|`kimm.models.MobileNetV2*`|
|MobileNetV3|[ICCV 2019](https://arxiv.org/abs/1905.02244)|`timm`|`kimm.models.MobileNetV3*`|
|MobileViT|[ICLR 2022](https://arxiv.org/abs/2110.02178)|`timm`|`kimm.models.MobileViT*`|
|RegNet|[CVPR 2020](https://arxiv.org/abs/2003.13678)|`timm`|`kimm.models.RegNet*`|
|ResNet|[CVPR 2015](https://arxiv.org/abs/1512.03385)|`timm`|`kimm.models.ResNet*`|
|TinyNet|[NeurIPS 2020](https://arxiv.org/abs/2010.14819)|`timm`|`kimm.models.TinyNet*`|
|VGG|[ICLR 2015](https://arxiv.org/abs/1409.1556)|`timm`|`kimm.models.VGG*`|
|ViT|[ICLR 2021](https://arxiv.org/abs/2010.11929)|`timm`|`kimm.models.VisionTransformer*`|
|Xception|[CVPR 2017](https://arxiv.org/abs/1610.02357)|`keras`|`kimm.models.Xception`|

The export scripts can be found in `tools/convert_*.py`.

## License

Please refer to [timm](https://github.com/huggingface/pytorch-image-models#licenses) as this project is built upon it.

### `kimm` Code

The code here is licensed Apache 2.0.

## Acknowledgements

Thanks for these awesome projects that were used in `kimm`

- [https://github.com/keras-team/keras](https://github.com/keras-team/keras)
- [https://github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)

## Citing

### BibTeX

```bash
@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
}
```

```bash
@misc{hy2024kimm,
  author = {Hongyu Chiu},
  title = {Keras Image Models},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/james77777778/kimm}}
}
```

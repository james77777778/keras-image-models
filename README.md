<!-- markdownlint-disable MD033 -->
<!-- markdownlint-disable MD041 -->

# Keras Image Models

<div align="center">
<img width="50%" src="docs/banner/kimm.png" alt="KIMM">

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

### Use Pretrained Model

```python
from keras import random

import kimm

# Use `kimm.list_models` to get the list of available models
print(kimm.list_models())

# Specify the name and other arguments to filter the result
print(kimm.list_models("efficientnet", weights="imagenet"))  # fuzzy search

# Initialize the model with pretrained weights
model = kimm.models.EfficientNetV2B0(weights="imagenet")

# Predict
x = random.uniform([1, 192, 192, 3]) * 255.0
y = model.predict(x)
print(y.shape)

# Initialize the model as a feature extractor with pretrained weights
model = kimm.models.EfficientNetV2B0(
    feature_extractor=True, weights="imagenet"
)

# Extract features for downstream tasks
y = model.predict(x)
print(y.keys())
print(y["BLOCK5_S32"].shape)
```

### Transfer Learning

```python
from keras import layers
from keras import models
from keras import random

import kimm

# Initialize the model as a backbone with pretrained weights
backbone = kimm.models.EfficientNetV2B0(
    input_shape=[224, 224, 3],
    include_top=False,
    pooling="avg",
    weights="imagenet",
)

# Freeze the backbone for transfer learning
backbone.trainable = False

# Construct the model with new head
inputs = layers.Input([224, 224, 3])
x = backbone(inputs, training=False)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(2)(x)
model = models.Model(inputs, outputs)

# Train the new model (put your own logic here)

# Predict
x = random.uniform([1, 224, 224, 3]) * 255.0
y = model.predict(x)
print(y.shape)
```

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

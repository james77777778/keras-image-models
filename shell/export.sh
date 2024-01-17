#!/bin/bash
set -Euxo pipefail

export CUDA_VISIBLE_DEVICES=
export TF_CPP_MIN_LOG_LEVEL=3
export KERAS_BACKEND=tensorflow
python3 -m tools.convert_densenet_from_timm
python3 -m tools.convert_efficientnet_from_timm
python3 -m tools.convert_ghostnet_from_timm
python3 -m tools.convert_inception_v3_from_timm
python3 -m tools.convert_mobilenet_v2_from_timm
python3 -m tools.convert_mobilenet_v3_from_timm
python3 -m tools.convert_mobilevit_from_timm
python3 -m tools.convert_regnet_from_timm
python3 -m tools.convert_resnet_from_timm
python3 -m tools.convert_vit_from_timm

echo "Export finished successfully!"

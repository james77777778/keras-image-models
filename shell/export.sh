#!/bin/bash
export CUDA_VISIBLE_DEVICES=
export TF_CPP_MIN_LOG_LEVEL=3
python3 -m tools.convert_densenet_from_timm &&
python3 -m tools.convert_efficientnet_from_timm &&
python3 -m tools.convert_ghostnet_from_timm &&
python3 -m tools.convert_inception_v3_from_timm &&
python3 -m tools.convert_mobilenet_v2_from_timm &&
python3 -m tools.convert_mobilenet_v3_from_timm &&
python3 -m tools.convert_mobilevit_from_timm &&
echo "Export finished successfully!"

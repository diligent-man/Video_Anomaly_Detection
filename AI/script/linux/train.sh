#!/bin/bash
# Should run with ./train.sh

# trained 7 epochs
python3 ../../src/tools/train.py --config ../../config/single/rgb_2D.json && \
python3 ../../src/tools/train.py --config ../../config/single/rgb_3D.json && \
python3 ../../src/tools/train.py --config ../../config/single/rgb_mixed.json


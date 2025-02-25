#!/bin/sh
python3 ../src/tools/preprocess_video.py --device both --processes 32 --root ../dataset/ubi-fight --save_root ../dataset/out
python3 ../src/tools/preprocess_video.py --device both --processes 32 --root ../dataset/iitb --save_root ../dataset/out

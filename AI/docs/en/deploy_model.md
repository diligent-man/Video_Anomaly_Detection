# Overview
In our project, we make such an assumption that both AI and Web services will be launched simultaneously, thus we synthesize all of them into a unified docker compose file for the ease of administration. At the moment, we provide three different versions of docker compose configurations, including [input_2d](../../../input_2d_compose.yaml), [input_3d](../../../input_3d_compose.yaml), [input_mixed](../../../input_mixed_compose.yaml), and you can run all of them at the same time.

## AI service
#### 1/ Prepare model
If model is from remote mlflow tracking server, please use [mlflow_export_import](../../script/linux/mlflow_export_import.sh) script to download appropriate experiment at first. You, then, modify appropriate path(s) for backbone's weight in downloaded configuration file. Check [this](./weight_info.md) for backbones' weight and our pretrained models.

#### 2/ Environment variable in Docker
There are four environment variables, be it `DEVICE`, `WEIGHT_PATH`, `PORT` and `CONFIG_PATH`, you should pay your attention to. `WEIGHT_PATH` is path to deployed model's weight with root from /weight. `CONFIG_PATH` is path to deployed model's configuration json file with root from /weight. `PORT` is used in case of port conflict circumstance.

## BE service
#### 1/ TMP_DIR environment variable
This variable is used to determine root directory for `plots`, `scores` and `videos`, which are used to store inferred results.

#### 2/ MERGE_GAP environment variable
Two anomalous regions are merged if their frame distance ≤ `MERGE_GAP`.

#### 3/ VAD_ENDPOINT environment variable
The endpoint of AI service.

#### 4/ Smoother configuration
We applied Savitzky-Golay filter to smooth input anomalous score, which requires `WINDOW_LENGTH` and `WINDOW_LENGTH` parameter. See more at [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html#scipy.signal.savgol_filter). 

#### 5/ Find_peak configuration
We utilized `find_peak()` from scipy to detect anomalous regions from model's projected scores. See more at [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html).

## FE service
#### 1/ VITE_API_URL environment variable
The endpoint of BE service.

#### 2/ VITE_PORT environment variable
The exposed port of FE service.

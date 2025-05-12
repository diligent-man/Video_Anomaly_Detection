# Data Preprocessing
Our preprocessing pipeline is crytalized in [VideoPreprocessor](../../src/preprocessing/VideoPreprocessor.py) module, and it contains two different stages. We create scripts for both Linux and Windows OS, but below is an exposition when running on Linux. 
* Stage 1  
Resampling video with specified FPS → Rescale → Central crop -> Extract video stream ([script](../../script/linux/preprocess_stage1.sh)).
* Stage 2:  
Split video into n non-overlapping video segments (clips) -> Temporal sampling -> Save as Pytorch tensor (.pt format) ([script](../../script/linux/preprocess_stage2.sh)).
* Stage 3:  
Copy remaining files to preprocessed directory ([script](../../script/linux/preprocess_stage3.sh)).

### Common configurations
| Parameter    | Default                                | Description                                                                                         |
|--------------|----------------------------------------|-----------------------------------------------------------------------------------------------------|
| `device`     | <p align=center> `"cpu"`               | Device for preprocessing video with ffmpeg. Note that ffmpeg should be built with GPU acceleration. |
| `cpu_ratio`  | <p align=center> `0.5`                 | Ratio b/t the utilization of cpu and gpu if device is both.                                         |
| `save_root`  | <p align=center> `"preprocessed"`      | Output root of preprocessed videos.                                                                 |
| `root`       | <p align=center> `--`                  | Root of dataset, which is read by VideoFolderDataset class.                                         |
| `loader`     | <p align=center> `"v6"`                | Video loader api ([check this](../../src/utils/load_video.py)).                                     |
| `batch_size` | <p align=center> `48`                  | Dataloader batch size.                                                                              |
| `processes`  | <p align=center> `os.cpu_count() // 2` | Number of processes for multiprocessing.                                                            |
| `fn_name`    | <p align=center> `--`                  | Which preprocessing stage to run.                                                                   |

### Stage 1 only
| Parameter    | Default                  | Description                                   |
|--------------|--------------------------|-----------------------------------------------|
| `run_async`  | <p align=center> `False` | Waiting time when running in an async manner. |
| `wait_time`  | <p align=center> `20`    | Run ffmpeg in an async manner.                |


### Stage 2 only  
| Parameter                  | Default                           | Description                                                                        |
|----------------------------|-----------------------------------|------------------------------------------------------------------------------------|
| `del_prev_result`          | <p align=center> `False`          | Delete previous stage result.                                                      |
| `include_labeled`          | <p align=center> ``               | Also split labeled video into segements that will be used for train/ val later on. |

### Stage 3 only  
| Parameter | Default               | Description                                                                   |
|-----------|-----------------------|-------------------------------------------------------------------------------|
| `vid_ext` | <p align=center> `--` | List of video extension string for being ignored when moving remaining files. |

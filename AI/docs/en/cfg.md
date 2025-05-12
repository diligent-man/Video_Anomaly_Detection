# Configuration overview
Our configuration file is responsible for managing 7 important fields, namely global, data, architecture, etc., and these settings can substantially affect the model's behavior at various stages, including training and testing.

# Global
| Parameter            | Default                                     | Description                                                                                                                                          |
|----------------------|---------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| `save_dir`           | <p align=center> `"results"`                | Final save dir is joined from <save_dir>, <project_name>, <experiment_name>, <technique>, <mode> and <experiment_name>.                              |
| `project_name`       | <p align=center> `"nameless_project"`       | <p align=center> See save_dir arg.                                                                                                                   |
| `experiment_name`    | <p align=center> `"run"`                    | <p align=center> See save_dir arg.                                                                                                                   |
| `technique`          | <p align=center> `"single"`                 | <p align=center> See save_dir arg.                                                                                                                   |
| `mode`               | <p align=center> `"train"`                  | See save_dir arg. Currently accept "train" or "test".                                                                                                |
| `exist_ok`           | <p align=center> `False`                    | See save_dir arg. If False, incremented version will be created, else reused old directory.                                                          |
| `sep`                | <p align=center> `""`                       | See save_dir arg. Seperator (sep) is added to experiment_name when exist_ok=False. Sep can be any ascii character.                                   |
| `epoch`              | <p align=center> `1`                        | Number of epoch to train                                                                                                                             |
| `eval_strategy`      | <p align=center> `"epoch"`                  | Whether or not evaluate model during training. Currently, we support "no" \| "epoch" \| "step".                                                      |
| `eval_steps`         | <p align=center> `lenght of val dataloader` | Number of step to perform evaluation in case of eval_strategy="step".                                                                                |
| `resume`             | <p align=center> `False`                    | Continue to train old model or not.                                                                                                                  |
| `resume_ckpt`        | <p align=center> `""`                       | Path to checkpoint for resuming.                                                                                                                     |
| `use_amp`            | <p align=center> `False`                    | Apply mixed precision training or not. More details at [this](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html). |
| `seed`               | <p align=center> `n/a`                      | Currently, we did not implement for the use of this Parameter.                                                                                        |
| `device`             | <p align=center> `"cpu"`                    | Device to run model on. We just fully experimented with "cpu" and "cuda".                                                                            |
| `inspect_model_arch` | <p align=center> `False`                    | Retrieve model architecture by torchinfo or not. Retrieved info is automatically logged out in conjunction with mlflow service.                      |
| `inspect_depth`      | <p align=center> `3`                        | Depth of nested layers to display (e.g. Sequentials). Nested layers below this depth will not be displayed in the summary.                           |
| `dummy_input_shape`  | <p align=center> `None`                     | Shape of dummy input tensor for running torchinfo.                                                                                                   |

# Data
Currently, dataset building process only accept "train" | "val" | "test" mode. Each process requires config for dataset, dataloader, and forward_strategy.
### dataset 
| Parameter           |  Default              | Description                                                                                                                                                                                                          |
|---------------------|-----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `name`              | <p align=center> --   | Name of dataset. Each dataset is inherited from torch.utils.data.Dataset [source](https://github.com/pytorch/pytorch/blob/v2.7.0/torch/utils/data/dataset.py#L39). See at [this](../../src/data/dataset/__init__.py) |
| `transform`         | <p align=center> None | Required field for each dataset implementation. Transform is applied on read input data. More details at [this](../../src/data/transform/__init__.py)                                                                |
| `target_transform`  | <p align=center> None | Required field for each dataset implementation. Target transform is applied on read label data. More details at [this](../../src/data/transform/__init__.py)                                                         |
| `kwarg`             | <p align=center> ---  | Other arguments required for building dataset. Detailed information is at each dataset.                                                                                                                              |

### dataloader
| Parameter   | Default              | Description                                                                                                                                                                 |
|-------------|----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `name`      | <p align=center> --  | Name of dataloader. Each dataloader is inherited from torch.utils.data.Dataset [source](https://github.com/pytorch/pytorch/blob/v2.7.0/torch/utils/data/dataloader.py#L131) |
| `kwarg`     | <p align=center> --  | Other arguments required for building dataloader. Details at [this](../../src/data/dataloader/__init__.py)                                                                  |

### forward_strategy 
| Parameter          | Default                        | Description                                                        |
|--------------------|--------------------------------|--------------------------------------------------------------------|
| `forward_strategy` | <p align=center> `--`          | Name of loop that would be used in training or testing phase.      |
| `overridden_args`  | <p align=center> `DotDict({})` | Arguments to be overriden in specified forward_strategy function.  |

# Architecture
Our implementations consist of two different types of model architecture: single model architecture (called one-stage training) and distilled model architecture (called two-stage training). The base class for build model is [BaseModel](../../src/modeling/architectures/BaseModel.py) which splits building process into four stages: build_backbone, build_neck, build_head and build_postprocessing. Each stage can be left blank if necessary.  
### One-stage training (a.k.a Teacher model)
| Parameter    | Default                     | Description                                                                                                      |
|--------------|-----------------------------|------------------------------------------------------------------------------------------------------------------|
| `algorithm`  | <p align=center> `"single"` | Name of loop that would be used in training or testing phase. Currently, we support "single" and "distillation". |
| `backbone`   | <p align=center> `--`       | Corresponding kwarg for building backbone. More details at [this](../../src/modeling/backbones/__init__.py).     |
| `neck`       | <p align=center> `--`       | Corresponding kwarg for building backbone. More details at [this](../../src/modeling/necks/__init__.py).         |
| `head`       | <p align=center> `--`       | Corresponding kwarg for building backbone. More details at [this](../../src/modeling/heads/__init__.py).         |


### Two-stage training (a.k.a Student model)
In this two-stage training, we support feature-based distillation algorithm at the moment.

| Parameter              | Default                     | Description                                                                                                                                                                                                                                                                                                                 |
|------------------------|-----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `algorithm`            | <p align=center> `"single"` | <p align=center> See one-stage training.                                                                                                                                                                                                                                                                                    |
| `soft_label_threshold` | <p align=center> `0.5`      | Threshold for converting teacher model's prediction into pseud-labels that are used in calculating distillation loss later on.                                                                                                                                                                                              |
| `feat_preprocessing`   | <p align=center> `None`     | Preprocess features extracted in backbone stage. Currently, we support [MLP](../../src/modeling/nn/MLP.py) module.                                                                                                                                                                                                          |
| `models.student`       | <p align=center> `--`       | Configuration for multiple student models and ordinal order should be awared for computing distillation loss later on (e.g. [CombinedLoss](../../src/losses/CombinedLoss.py)). Each student model is built based on [BaseModel](../../src/modeling/architectures/BaseModel.py) with configuration as in one-stage training. |
| `models.teacher`       | <p align=center> `--`       | Configuration for multiple teacher models and ordinal order should be awared for computing distillation loss later on (e.g. [CombinedLoss](../../src/losses/CombinedLoss.py)). Each student model is built based on [BaseModel](../../src/modeling/architectures/BaseModel.py) with configuration as in one-stage training. |         

# Optimizer

| Parameter         | Default                        | Description                                                                                                    |
|-------------------|--------------------------------|----------------------------------------------------------------------------------------------------------------|
| `lr.name`         | <p align=center> `None`        | Name of optimizer used for training. Currently, we support all optimizers from PyTorch.                        |
| `lr.kwarg`        | <p align=center> `DotDict({})` | Corresponding Parameter to selected optimizer. See more at [here](../../src/optimizer/optimizer/__init__.py)    |
| `scheduler.name`  | <p align=center> `None`        | Name of learning scheduler for training. See more at [this](../../src/optimizer/lr_scheduler/__init__.py)      |
| `scheduler.kwarg` | <p align=center> `DotDict({})` | Corresponding Parameter to selected optimizer. See more at [here](../../src/optimizer/lr_scheduler/__init__.py) |                                                                                                                                                                                                                                                                                                                            |
| `regularizer`     | <p align=center> `--`          | Not developed !                                                                                                |


# Metric
| Parameter    | Default                  | Description                                                                                                                                                 |
|--------------|--------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `in_train`   | <p align=center> `False` | Calculate metrics during train or not.                                                                                                                      |
| `in_val`     | <p align=center> `False` | Calculate metrics during val or not.                                                                                                                        |
| `in_test`    | <p align=center> `False` | Calculate metrics during test or not.                                                                                                                       |
| `metrics`    | <p align=center> `[]`    | "name" and other corresponding kwarg for selected metric. Required to specify at least 1 metric regardless of value of `in_train` or `in_val` or `in_test`. |

# Loss
| Parameter  | Default                 | Description                                                                                                                                                  |
|------------|-------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `name`     | <p align=center> `None` | Name of loss used during training                                                                                                                            |
| `kwarg`    | <p align=center> `{}`   | Corresponding kwargs for selected loss. More details at [__LOSSES](../../src/losses/LossWrapper.py) field of [LossWrapper](../../src/losses/LossWrapper.py)  |


# Services
At the moment, we offer services (also called callbacks) as below:
* Tester: DefaultFlow (non-configurable).
* Trainer: DefaultFlow (non-configurable), Progress (non-configurable), Checkpointer (configurable), EarlyStopper (configurable).
* Integrated service: Mlflow (configurable, only Trainer).

### Checkpointer
| Parameter                 | Default                       | Description                                                                                                       |
|---------------------------|-------------------------------|-------------------------------------------------------------------------------------------------------------------|
| `save_dir`                | <p align=center> `--`         | Directory for saving satisfied checkpoints during training.                                                       |
| `mode`                    | <p align=center> `"min"`      | How best value is updated. Currently, we offer `"min"`.                                                           |
| `monitor`                 | <p align=center> `"val_loss"` | Monitored value to perform saving. Currently, `"val_loss"` is supported.                                          |
| `save_freq`               | <p align=center> `"epoch"`    | Saving frequency. It could be `"epoch"` for saving each epoch or an integer for saving after each i iterations.   |
| `save_best_only`          | <p align=center> `False`      | Only saves when the model is considered the "best" and the latest best model according to the monitored quantity. |
| `save_weights_only`       | <p align=center> `False`      | Only save model's weights.                                                                                        |
| `save_total_limit`        | <p align=center> `5`          | Total number of checkpoints will be stored during training.                                                       |
| `include_config`          | <p align=center> `False`      | Include config into saved checkpoint or not.                                                                      |
| `initial_value_threshold` | <p align=center> `None`       | Floating point initial "best" value of the metric to be monitored.                                                |
| `verbose`                 | <p align=center> `True`       | Prompt monitored quantity wordily.                                                                                |

### Earlystopper
| Parameter          | Default                       | Description                                                                |
|--------------------|-------------------------------|----------------------------------------------------------------------------|
| `mode`             | <p align=center> `"min"`      | How best value is evaluated to early stop. Currently, we offer `"min"`.    |
| `monitor`          | <p align=center> `"val_loss"` | Monitored quantity for evaluating early stop condition.                    |
| `patience`         | <p align=center> `5`          | Number of epochs with no improvement after which training will be stopped. |
| `min_delta`        | <p align=center> `0.`         | Additional boundary when evaulating early stop condition.                  |
| `check_from_epoch` | <p align=center> `0`          | Start early since i<sup>th</sup> epoch.                                    |
| `verbose`          | <p align=center> `True`       | Prompt result wordily.                                                     |

### Mlflow
| Parameter             | Default                  | Description                                                                                                |
|-----------------------|--------------------------|------------------------------------------------------------------------------------------------------------|
| `save_dir`            | <p align=center> `--`    | Directory for saving mlflow experiment.                                                                    |
| `prev_run_id`         | <p align=center> `None`  | Continue to use previous experiment, especially useful for resuming training.                              |
| `init_server_on_run`  | <p align=center> `False` | Spin up local mlflow server when start training.                                                           |
| `username`            | <p align=center> `None`  | Username for authenticating remote tracking server. If None, it retrieves value from environment variable. |
| `password`            | <p align=center> `None`  | Password for authenticating remote tracking server. If None, it retrieves value from enviroment variable.  |
| `push_to_remote`      | <p align=center> `False` | Automatically push experiment from local mlflow server to remote server.                                   |
| `remote_tracking_uri` | <p align=center> `None`  | Uri of remote tracking server.                                                                             |

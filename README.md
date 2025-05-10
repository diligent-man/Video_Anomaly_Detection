# Anomalous Human Activity Detection By Weakly Supervised Learning
## What is project for
* Implement training, testing and inference workflow for [VAD](https://paperswithcode.com/task/anomaly-detection-in-surveillance-videos) problem.
* Apply four (4, 5, 6, 7) out of nine MLOps principles ([FIGURE 2](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10081336)). 

## Repository organization
For the ease of source code management, this repo encompasses all implemented code and relatable configurations, and they are placed in AI, MLops and Web directories.
[AI](./AI) includes source code for building DL-based VAD algorithm. [Web](./Web) comprises both backend and frontend for simple demo web app. [MLOps](./MLOps) respectively consists of Docker configurations for lakeFS, minIO, MLflow, postgress services.

## Team members:
* Nguyễn Đức Trọng - SE160931
  * GitHub: [click here](https://github.com/diligent-man)
  * Role: Team leader
  * Tasks:  
    * Set up and manage MLOps-related services [here](./MLOps).  
    * Propose solution and implement core modules in [AI](./AI/).  
    * Plan project management document [here](./PMP.mpp)  
    * Review code.
* Cao Khánh Vy - SE162136  
  * GitHub: [here](https://github.com/vy11244)
  * Role: Team member
  * Tasks:  
    * Seek and make dataset.  
    * Version dataset with LakeFS service.
* Nguyễn Thế Hoàng - SE170420
  * GitHub: [here](https://github.com/Hoangnt1209)
  * Role: Team member
  * Tasks:  
    * Support in making dataset.
    * Implement prototype web.
* Nguyễn Ngọc Chiến - SE173133
  * GitHub: [here](https://github.com/chiennguyen383)
  * Role: Team member
  * Tasks:
    * Support in implementing code in [AI](./AI).
    * Support in finding and creating dataset.

## ✨ Key Features That Might Render You Enthralling Or Not:
* ⚡ **Easy-to-configure model [Trainer](./AI/src/runner/Trainer.py) and [Tester](AI/src/runner/Tester.py)**: Quickly training and testing DL-based VAD model with json and yaml-supported configuration.  
* ⚙ **Dotted dictionary ([DotDict](./AI/src/utils/DotDict.py))**: Simple yet effective dotted dictionary with recursive approach for managing configuration in [AI](./AI).
* ⚙ **Multiple backbones handling ([MultiBackboneForwarder](./AI/src/modeling/backbones/MultiBackboneForwarder.py))**: Synthesize and perform forward pass upon multiple backbones in online manner (check []()).
* ⚙ **Scalable training and testing code ([forward_strategy](./AI/src/utils/forward_strategy.py))**: Easy to insert new running loop for training or testing. 
* ⚙ **Callback support ([check](./AI/src/callbacks/))**: Currently, we just support fundamental callbacks for trainer, tester and mlflow.
* ⚙ **Backbone-Neck-Head perspective ([BaseModel](./AI/src/modeling/architectures/BaseModel.py))**: Dissect and view model architecture in YOLO-like fashion.
* 🖥️ **Intuitive demo web:** User friendly demo web interface.

## 🚀 Get Started:
### AI Installation
please check at [here](./AI/README.md)
### MLOps Installation
please check at [here](./MLOps/README.md)
### Web Installation
please check at [here](./Web/README.md)

##  Acknowledgements:
We want to send our gratitude to following repos due to their inspiration for our implementation in [AI](./AI):  
* [Box](https://github.com/cdgriffith/Box)
* [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
* [transformers](https://github.com/huggingface/transformers)
* [mlflow-export-import](https://github.com/mlflow/mlflow-export-import)
* [Ultralytics](https://github.com/ultralytics/ultralytics) (a.k.a YOLO)
* [vision](https://github.com/pytorch/vision) (part of PyTorch project)

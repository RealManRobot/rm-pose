# FoundationPose_SDK

## **1. 项目介绍**

该模型可以根据物品mask图和CAD的3D模板来对物品进行位姿估计。它的输入是一张RGB图像和一张深度图、一张mask、CAD模板以及相机的内参，输出的是物品的姿态。
它不需要训练，使用提供的权重即可。
有了它，可以在实际对物品抓取如新零售等场景进行落地。

- **API链接**：[API链接地址](http://192.168.0.188:8090/ai_lab_rd02/ai_sdks/foundationpose.git)

## **2. 代码结构**

```
Foundationpose/
│
├── README.md        <- 项目的核心文档
├── requirements.txt    <- 项目的依赖列表
├── setup.py        <- 项目的安装脚本
│
├── FoundationPose/     <- 项目的源代码
│  ├── config           <- yaml配置文件夹
│  ├── debug            <- debug日志入口
│  ├── kaolin           <- 渲染功能依赖库
│  ├── learning         <- 数据处理与模型模块
│  ├── mycpp            <- 依赖库
│  ├── nvdiffrast       <- 高性能渲染依赖库
│  ├── datareader.py    <- 预测数据读取功能
│  ├── estimater11.py   <- 评估函数
│  ├── Utils.py         <- 解码部分
│  └── foundationpose_main.py    <- 核心功能接口函数

├── predict.py       <- 预测程序的主入口

└── tests/      <- 功能测试目录

```

## **3.环境与依赖**

* python3.8+
* torch==2.0.0+cu118
* torchvision==0.15.1+cu118
* torchaudio==2.0.1+cu118
* numpy
* opencv-python

## **4. 安装说明**

1. 安装Python 3.8或者更高版本
2. 克隆项目到本地：``
3. 进入项目目录：``
4. 安装依赖：
    - conda create -n foundationpose python=3.8
    - conda activate foundationpose
    - pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
    - pip install pyrealsense2
    - pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" # 若编译失败，应该是内存不足或系统资源耗尽引起的，可以先手动编译，限制编译时使用的并行工作线程数量 export MAX_JOBS=4 python setup.py bdist_wheel
    - pip install scipy joblib scikit-learn ruamel.yaml trimesh pyyaml opencv-python imageio open3d transformations warp-lang einops kornia pyrender pysdf
    - pip install git+https://github.com/facebookresearch/segment-anything.git
    - git clone https://github.com/NVlabs/nvdiffrast
    - cd nvdiffrast && pip install .
    - pip install scikit-image meshcat webdataset omegaconf pypng Panda3D simplejson bokeh roma seaborn pin opencv-contrib-python openpyxl torchnet Panda3D bokeh wandb colorama GPUtil imgaug Ninja xlsxwriter timm albumentations xatlas rtree nodejs jupyterlab objaverse g4f ultralytics==8.0.120 pycocotools py-spy pybullet videoio numba
    - pip install -U git+https://github.com/lilohuang/PyTurboJPEG.git
    - conda install -y -c anaconda h5py
    - sudo apt-get install libeigen3-dev -y
    - sudo apt-get install pybind11-dev -y
    - sudo apt-get install libboost-all-dev -y
    - cd FoundationPose/ && bash build_all.sh
    - git clone https://github.com/NVIDIAGameWorks/kaolin.git
    - cd kaolin/
    - git switch -c v0.15.0
    - pip install -e .
    - cd ..
    
5. 编译打包：在与 `setup.py `文件相同的目录下执行以下命令：`python setup.py bdist_wheel`。 在 `dist` 文件夹中找到 `.wheel` 文件，例如：`dist/foundationpose-0.1.0-py3-none-any.whl`。
6. 安装：`pip install foundationpose-0.1.0-py3-none-any.whl`

## **5. 使用指南**

### 推荐硬件&软件&环境配制

- 3090Ti对应安装nvidia-driver-535
- 安装对应cuda版本11.4及以上
- 安装对应torch和torchvision版本
- 命令：`pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118`


### 如何配置权重

`foundationpose_weights`：共2个权重

下载地址：https://alidocs.dingtalk.com/i/nodes/mExel2BLV5R5AEzpIYQkz77qJgk9rpMq?utm_scene=team_space

## 6. 接口示例

```python
import copy
import json
import os.path
from types import SimpleNamespace
import cv2
import pyrealsense2 as rs
from FoundationPose.estimater11 import *
from FoundationPose.datareader import *
from FoundationPose.foundationpose_main import Detect_foundationpose


def load_resources(mesh_path, intrinsics, predict_ckpt_dir, refine_ckpt_dir):
    est, reader, bbox, debug, to_origin = Detect_foundationpose.load_model(mesh_path, intrinsics, predict_ckpt_dir, refine_ckpt_dir)
    return est, reader, bbox, debug, to_origin


def main():
    image_path = "tests/demo_data/test_img"
    color_path = os.path.join(image_path, "rgb.png")
    depth_path = os.path.join(image_path, "depth.png")
    mask_path = os.path.join(image_path, "mask.png")

    mesh_path = "tests/demo_data/haoliyou/mesh/textured_mesh.obj"

    predict_ckpt_dir = "tests/weights/predict_ckpt/predict_ckpt.pth"
    refine_ckpt_dir = "tests/weights/refine_ckpt/refine_ckpt.pth"

    json_file = os.path.join(image_path, "intrinsics.json")
    with open(json_file, 'r+') as fp:
        intrinsics = json.load(fp, object_hook=lambda d: SimpleNamespace(**d))

    color_img = cv2.imread(color_path)
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 调用加载mesh和相关的资源
    est, reader, bbox, debug, to_origin = load_resources(mesh_path, intrinsics, predict_ckpt_dir, refine_ckpt_dir)
    # 根据mesh进行位姿估计
    pose, color, to_origin = Detect_foundationpose.pose_est(color_img, depth_img, mask, reader, est,
                                                            to_origin, bbox,
                                                            show=True)
    # 位姿估计的可视化
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    cv2.imshow("pose", color)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()

```

## 7. **许可证信息**

说明项目的开源许可证类型（如MIT、Apache 2.0等）。

* 本项目遵循MIT许可证。

## 8. 常见问题解答（FAQ）**

列出一些常见问题和解决方案。

- **Q1：机械臂连接失败**

  答案：修改过机械臂IP

- **Q2：UDP数据推送接口收不到数据**

  答案：检查线程模式、是否使能推送数据、IP以及防火墙

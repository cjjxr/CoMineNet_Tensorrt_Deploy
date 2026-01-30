# CoMineNet C++ Deployment (CoMineNet_Deploy)

这是一个基于 C++、CUDA 和 TensorRT 的高性能部署项目，集成了 **CoMineNet** 目标检测、可行驶区域分割模型和 **DepthAnything** 深度估计模型。项目旨在 Jetson 等边缘计算设备或高性能 GPU 服务器上实现实时的目标检测与距离测量。
![Demo](assets/demo.mp4)

## 📁 目录结构

项目采用标准的 C++ 工程结构：

```
CoMineNet_cpp_deploy/
├── src/                # 源代码文件
│   ├── main.cu         # 程序入口，负责配置加载和主循环
│   ├── DepthAnything.cu # DepthAnything 模型推理实现
│   └── JetsonDetector.cu # CoMineNet 检测模型推理实现
├── include/            # 头文件
│   ├── DepthAnything.hpp
│   └── JetsonDetector.hpp
├── config/             # 配置文件
│   └── config.yaml     # 项目参数配置文件
├── models/             # 模型文件存放目录 (需自行放入 .engine 文件)
├── CMakeLists.txt      # CMake 构建脚本
└── README.md           # 项目说明文档
```

## 🛠️ 环境依赖

在编译和运行本项目之前，请确保已安装以下依赖库：

*   **CMake** (>= 3.18)
*   **C++ Compiler** (支持 C++17)
*   **CUDA Toolkit** (推荐 11.x 或 12.x)
*   **TensorRT** (与 CUDA 版本兼容)
*   **OpenCV** (4.x)
*   **yaml-cpp** (用于读取 YAML 配置文件)

## 🚀 编译与构建

1.  **克隆项目**
    ```bash
    git clone <repository_url>
    cd CoMineNet_cpp_deploy
    ```

2.  **创建构建目录**
    ```bash
    mkdir build && cd build
    ```

3.  **运行 CMake**
    ```bash
    cmake ..
    ```

4.  **编译**
    ```bash
    make -j$(nproc)
    ```

## ⚙️ 配置说明

项目使用 `config/config.yaml` 进行参数配置。在运行前，请确保配置文件中的路径和参数正确。

```yaml
common:
  input_dir: "./data/input"   # 输入图片文件夹路径或视频文件路径 (支持 mp4, avi 等)
                              # 如果输入数字 (如 "0")，则尝试打开摄像头
  output_dir: "./data/output" # 结果保存目录

rmt_ppad:
  model_path: "./models/CoMineNet.engine" # 目标检测 TensorRT 引擎路径
  input_width: 640
  input_height: 640

depth_anything:
  model_path: "./models/depth_anything.engine" # 深度估计 TensorRT 引擎路径
  input_width: 518
  input_height: 518
```

## 🏃 运行说明

1.  **准备模型文件**
    将转换好的 TensorRT 引擎文件 (`.engine`) 放置在 `models/` 目录下（或修改配置文件指向正确路径）。

2.  **准备输入数据**
    在 `data/input` 目录下放入测试图片，或在配置文件中指定视频路径/摄像头索引。

3.  **执行程序**
    在 `build` 目录下运行：
    ```bash
    ./rmt_deploy
    ```
    程序将自动读取 `../config/config.yaml`（或 `./config.yaml`）并开始处理。

## ✨ 主要功能

*   **多模型协同**：同时运行目标检测和单目深度估计。
*   **CUDA 加速**：
    *   **预处理**：使用 CUDA Kernel 实现 Resize、归一化和格式转换 (HWC -> CHW)。
    *   **后处理**：GPU 端处理推理结果。
*   **实时测距**：结合检测框和深度图，计算目标的物理距离。
*   **灵活输入**：支持图片文件夹、视频文件和实时摄像头流。

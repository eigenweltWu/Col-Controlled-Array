# Col-Controlled-Array-1-bit-1-control

每列由一条DC线统一控制，初始相位和相位偏移均为1-bit。

运行`run_global_optimization.py`计算最优初始相位，运行`run_post_processing.py`对给定的最优初始相位求解密码本。

环境部署依赖conda，请运行`conda create -f dataprocess_win.yml`在windows上部署，运行`conda env create -f dataprocess_ubuntu2404.yml`在Ubuntu 24.04上部署。

如需使用GPU加速（强烈推荐），该环境依赖Cuda12.6，因而请确保您的NVIDIA驱动版本能够支持（Win≥560.76, Linux≥560.28.03），并从https://developer.nvidia.com/cuda-12-6-0-download-archive 下载并安装CUDA。

### `run_global_optimization`典型计算时长（注：$M,N\le 16$与$M,N>16$求解算法有差别）：

|M|N|n_generation|population_size|i7-14700+RTX 4090D|i5-14600kf+RTX 3060|E5-2620v3|i7-14700|Memory(MB)|
|---|---|---|---|---|---|---|---|---|
|14|14|50|80|5m52s|12m32s|3h56m12s|1h6m41s|1708|
|14|14|120|200|34m46s|1h13m41s|N/A|N/A|3822|
|31|31|300|200|1h18m11s|2h43m23s|N/A|N/A|3712|

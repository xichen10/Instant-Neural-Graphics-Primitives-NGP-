下载050-Instant-Neural-Graphics-Primitives-with-a-Multiresolution-Hash-Encoding-main.zip，解压。这是NGP的训练代码，python train.py能跑起来就算成功。

第一步，环境搭建
  安装gcc，服务器默认已安装，通过gcc --version查看版本号
  安装CUDA，服务器默认已安装，通过nvcc --version查看版本号
  安装python3（推荐python3.8），服务器默认已安装，通过python3 --version查看版本号
  安装anaconda（官网下载，wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh，安装，bash Anaconda3-2023.03-Linux-x86_64.sh）
  创建虚拟环境（conda create -n xxx python=xxx)

  安装各种库（大部分直接pip install）
  pytorch（这里pytorch版本和CUDA版本要匹配）
  pytorch_lightning
    需指定版本号pip install pytorch-lightning==1.9.0，否则会报ImportError: cannot import name 'EPOCH_OUTPUT' from 'pytorch_lightning.utilities.types'，因为新版的pytorch-lightning移除了EPOCH_OUTPUT。这里旧版本安装巨快（1M左右），新版本巨慢（800M左右）。
  tinycudann（这个有点麻烦，参考https://blog.csdn.net/qq_45934285/article/details/140332600）
    安装tinycudann前先安装pytorch3d库，直接pip install pytorch3d
    下载tiny-cuda-nn-master.zip，解压。
    命令行cd tiny-cuda-nn-master/bindings/torch/
    命令行export TCNN_CUDA_ARCHITECTURES=70（这里为了告诉环境当前GPU的计算能力，不执行这条会报错EnvironmentError("Unknown compute capability.")，Tesla V100计算能力为7.0, A800的计算能力8.0？）
    命令行python setup.py install（可能会遇到cuda版本和gcc版本不一致的问题，同时记得完整fmt和cutlass）
  imageio
  cv2 (pip install opencv-python)
  einops
  kornia
  安装vren库（这里要cd models/csrc/, 运行python setup.py install）
  torch_scatter（https://zhuanlan.zhihu.com/p/504134665）
  apex（cd apex, python setup.py install --cpp_ext --cuda_ext）

第二步，数据集下载（https://github.com/facebookresearch/NSVF?tab=readme-ov-file#dataset）

第三步，开始训练
  python train.py --root_dir 指定数据集路径
  训练结果会保存到result/，模型会保存到ckpts/

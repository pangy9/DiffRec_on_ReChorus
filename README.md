# DiffRec_on_ReChorus

这是中山大学人工智能专业的大三机器学习大作业项目：在[ReChorus](https://github.com/THUwangcy/ReChorus)框架上复现一篇近期的顶会论文，我们选取的论文是[Diffusion Recommender Model](https://arxiv.org/abs/2304.04971)，代码参考了它的[公开仓库](https://github.com/YiyanXu/DiffRec?tab=readme-ov-file)

## 目录

- [安装与运行](#安装与运行)
- [致谢](#致谢)
- [许可证](#许可证)

---


## 安装与运行

### 依赖要求

我们的实验是在以下环境中进行的：

GPU 型号：单张 NVIDIA GeForce RTX 4090.

操作系统：Ubuntu 20.04; CUDA 版本：12.1; Python 版本：3.10.14.

创建虚拟环境后，运行
```
pip install -r requirements.txt
```

### 运行项目
进入`src`目录后，执行
```
bash my_new_scripts/DiffRec.sh
```

## 致谢

该项目建立在其他几个存储库的工作基础上。我们对以下项目的贡献表示感谢：

- [ReChorus](https://github.com/THUwangcy/ReChorus): “Chorus” of recommendation models: a light and flexible PyTorch framework for Top-K recommendation.
- [DiffRec](https://github.com/YiyanXu/DiffRec?tab=readme-ov-file): Implementation of Diffusion Recommender Model
---

## 许可证

本项目基于 [ReChorus](https://github.com/THUwangcy/ReChorus)，原始项目使用 [MIT 许可证](https://github.com/THUwangcy/ReChorus/blob/master/LICENSE)。 

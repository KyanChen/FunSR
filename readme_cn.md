# FunSR - 基于隐式函数空间中的上下文交互的连续遥感图像超分辨率

[中文]() [English]()

本项目是论文"Continuous Remote Sensing Image Super-Resolution based on Context Interaction in Implicit Function Space"的Pytorch实现


[项目主页](https://kyanchen.github.io/FunSR/) $\cdot$ [PDF下载](https://arxiv.org/abs/2302.08046) $\cdot$ [HuggingFace](https://huggingface.co/spaces/KyanChen/FunSR)


## 0. 环境准备

    0.1 建立虚拟环境：conda create -n FunSR python=3.10
    0.2 激活虚拟环境：conda activate FunSR
    0.3 安装pytorch （torch 2.x 建议）：pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
    0.3 [可选]安装pytorch：conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
    0.4 安装mmcv （mmcv 2.x 建议）：参考https://mmcv.readthedocs.io/zh_CN/latest/get_started/installation.html    
    0.5 安装其他依赖：pip install -r requirements.txt


## 1. 数据准备
  
    1.1 下载数据集：将下载的高分辨率图像放到*samples*文件夹中。在本项目中，该文件夹里面提供了一些示例图像。
    1.2 划分训练测试集：在本项目中已提供论文中的数据集划分文件，位于*data_split*文件夹中；如果需要自己划分，请使用*tools/data_tools/get_train_val_list.py*划分训练测试集, 运行`python tools/data_tools/get_train_val_list.py`。



## 2. 模型训练

    1.1 **FunSR模型**
    1.1.1 配置文件：configs/train_1x-5x_INR_funsr.yaml
    1.1.2 训练：python train_inr_funsr.py，可以依据情况修改该文件中的ArgumentParser参数

    1.2 [可选] 固定倍数超分模型（论文中的TransEnet, SRCNN, LGCNet, FSRCNN, DCM, VDSR）
    1.2.1 配置文件：configs/baselines/train_CNN.yaml
    1.2.2 训练：python train_cnn_sr.py，可以依据情况修改该文件中的ArgumentParser参数

    1.3 [可选] 连续倍数超分模型（论文中的LIIF, MetaSR, ALIIF）
    1.3.1 配置文件：configs/baselines/train_1x-5x_INR_[liif, metasr, aliif].yaml
    1.3.2 训练：python train_liif_metasr_aliff.py，可以依据情况修改该文件中的ArgumentParser参数

    1.4 [可选] 连续倍数超分模型（论文中的DIINN, ArbRCAN, SADN, OverNet）
    1.4.1 配置文件：configs/baselines/train_1x-5x_INR_diinn_arbrcan_sadn_overnet.yaml
    1.4.2 训练：python train_diinn_arbrcan_sadn_overnet.py，可以依据情况修改该文件中的ArgumentParser参数

## 3. 模型评测

    1.1 **FunSR模型**（同时包含论文中的DIINN, ArbRCAN, SADN, OverNet连续超分模型）
    1.1.1 配置文件：configs/test_INR_diinn_arbrcan_funsr_overnet.yaml
    1.1.2 测试：python test_inr_diinn_arbrcan_sadnarc_funsr_overnet.py，可以依据情况修改该文件中的ArgumentParser参数

    1.2 [可选] 插值超分方式（双线性，三线性）
    1.2.1 配置文件：configs/test_interpolate.yaml
    1.2.2 训练：python test_interpolate_sr.py，可以依据情况修改该文件中的ArgumentParser参数

    1.3 [可选] 固定倍数超分模型（论文中的TransEnet, SRCNN, LGCNet, FSRCNN, DCM, VDSR）
    1.3.1 配置文件：configs/test_CNN.yaml
    1.3.2 训练：python test_cnn_sr.py，可以依据情况修改该文件中的ArgumentParser参数

    1.4 [可选] 连续倍数超分模型（论文中的LIIF, MetaSR, ALIIF）
    1.4.1 配置文件：configs/test_INR_liif_metasr_aliif.yaml
    1.4.2 训练：python test_inr_liif_metasr_aliif.py，可以依据情况修改该文件中的ArgumentParser参数
    
    1.5 多分辨率评测
    1.5.1 为了评测不同超分倍数的便捷性，我们提供了一个批量评测的脚本，位于scripts/test_script.py中，可以运行`python scripts/test_script.py`

## 4. [可选]结果可视化

    4.1 本项目提供了一些可视化工具来产生论文中的可视化结果，位于tools/paper_vis_tools中，可以参考该文件夹中的文件了解详情

## 5. [可选]模型下载

    5.1 本项目提供了RDN的模型权重，位于[huggingface space](https://huggingface.co/spaces/KyanChen/FunSR/tree/main/pretrain)中

## 6. [可选]引用

    6.1 如果您认为本项目对您的研究有所帮助，请引用我们的论文：
    6.2 如果您有其他问题，请联系我！！！

    ```
    @article{chen2023continuous,
      title={Continuous Remote Sensing Image Super-Resolution based on Context Interaction in Implicit Function Space},
      author={Chen, Keyan and Li, Wenyuan and Lei, Sen and Chen, Jianqi and Jiang, Xiaolong and Zou, Zhengxia and Shi, Zhenwei},
      journal={IEEE Transactions on Geoscience and Remote Sensing},
      year={2023},
      publisher={IEEE}
    }
    ```


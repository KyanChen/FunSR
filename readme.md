# FunSR - Continuous Remote Sensing Image Super-Resolution based on Context Interaction in Implicit Function Space

[中文]() [English]()

This is the pytorch implement of our paper "Continuous Remote Sensing Image Super-Resolution based on Context Interaction in Implicit Function Space"


[Project Page](https://kyanchen.github.io/FunSR/) $\cdot$ [PDF Download](https://arxiv.org/abs/2302.08046) $\cdot$ [HuggingFace Demo](https://huggingface.co/spaces/KyanChen/FunSR)


## 0. Environment Setup
    
    0.1 Create a virtual environment: conda create -n FunSR python=3.10
    0.2 Activate the virtual environment: conda activate FunSR
    0.3 Install pytorch (torch 2.x is ok): pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
    0.3 [Optional] Install pytorch: conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
    0.4 Install mmcv (mmcv 2.x is ok): refer to https://mmcv.readthedocs.io/zh_CN/latest/get_started/installation.html
    0.5 Install other dependencies: pip install -r requirements.txt

## 1. Data Preparation
  
    1.1 Download the dataset: put the downloaded HR images into the *samples* folder. In this project, some example images are provided in this folder.
    1.2 Split the training and validation set: the data split files in the paper are provided in the *data_split* folder; if you need to split the training and validation set by yourself, please use *tools/data_tools/get_train_val_list.py* to split the training and validation set, run `python tools/data_tools/get_train_val_list.py`.


## 2. Model Training

    1.1 **FunSR**
    1.1.1 Config file: configs/train_1x-5x_INR_funsr.yaml
    1.1.2 Training: `python train_inr_funsr.py`, you can modify the ArgumentParser parameters in this file according to the situation.

    1.2 [Optional] Fixed-scale SR models (TransEnet, SRCNN, LGCNet, FSRCNN, DCM, VDSR)
    1.2.1 Config file: configs/baselines/train_CNN.yaml
    1.2.2 Training: `python train_cnn_sr.py`, you can modify the ArgumentParser parameters in this file according to the situation.

    1.3 [Optional] Continuous-scale SR models (LIIF, MetaSR, ALIIF)
    1.3.1 Config file: configs/baselines/train_1x-5x_INR_[liif, metasr, aliif].yaml
    1.3.2 Training: `python train_liif_metasr_aliff.py`, you can modify the ArgumentParser parameters in this file according to the situation.

    1.4 [Optional] Continuous-scale SR models (DIINN, ArbRCAN, SADN, OverNet)
    1.4.1 Config file: configs/baselines/train_1x-5x_INR_diinn_arbrcan_sadn_overnet.yaml
    1.4.2 Training: `python train_diinn_arbrcan_sadn_overnet.py`, you can modify the ArgumentParser parameters in this file according to the situation.

## 3. Model Evaluation

    1.1 **FunSR** (including DIINN, ArbRCAN, SADN, OverNet)
    1.1.1 Config file: configs/test_INR_diinn_arbrcan_funsr_overnet.yaml
    1.1.2 Testing: `python test_inr_diinn_arbrcan_sadnarc_funsr_overnet.py`, you can modify the ArgumentParser parameters in this file according to the situation.

    1.2 [Optional] Fixed-scale SR models (TransEnet, SRCNN, LGCNet, FSRCNN, DCM, VDSR)
    1.2.1 Config file: configs/baselines/test_CNN.yaml
    1.2.2 Testing: `python test_cnn_sr.py`, you can modify the ArgumentParser parameters in this file according to the situation.

    1.3 [Optional] Continuous-scale SR models (LIIF, MetaSR, ALIIF)
    1.3.1 Config file: configs/baselines/test_1x-5x_INR_[liif, metasr, aliif].yaml
    1.3.2 Testing: `python test_liif_metasr_aliff.py`, you can modify the ArgumentParser parameters in this file according to the situation.

    1.4 [Optional] Continuous-scale SR models (DIINN, ArbRCAN, SADN, OverNet)
    1.4.1 Config file: configs/baselines/test_1x-5x_INR_diinn_arbrcan_sadn_overnet.yaml
    1.4.2 Testing: `python test_diinn_arbrcan_sadn_overnet.py`, you can modify the ArgumentParser parameters in this file according to the situation.
    
    1.5 [Optional] Interpolation-based SR models (Bicubic, Bilinear)
    1.5.1 Config file: configs/test_interpolate.yaml
    1.5.2 Testing: `python test_interpolate_sr.py`, you can modify the ArgumentParser parameters in this file according to the situation.

## 4. [optional] Result Visualization

    4.1 Some visualization tools are provided in the *tools/paper_vis_tools* folder, you can refer to the files in this folder for details.

## 5. [optional] Model Download
    
    5.1 The model weights of RDN are provided in the [huggingface space](https://huggingface.co/spaces/KyanChen/FunSR/tree/main/pretrain)

## 6. [optional] Citation
    
    6.1 If you find this project useful for your research, please cite our paper:
    6.2 If you have any other questions, please contact me!!!
        
    ```
    @article{chen2023continuous,
      title={Continuous Remote Sensing Image Super-Resolution based on Context Interaction in Implicit Function Space},
      author={Chen, Keyan and Li, Wenyuan and Lei, Sen and Chen, Jianqi and Jiang, Xiaolong and Zou, Zhengxia and Shi, Zhenwei},
      journal={IEEE Transactions on Geoscience and Remote Sensing},
      year={2023},
      publisher={IEEE}
    }
    ```



















# FunSR

Official Implement of the Paper "Continuous Remote Sensing Image Super-Resolution based on Context Interaction in Implicit Function Space"

## Some Information

[Project Page](https://kyanchen.github.io/FunSR/) $\cdot$ [PDF Download](https://arxiv.org/abs/2302.08046) $\cdot$ [HuggingFace Demo](https://huggingface.co/spaces/KyanChen/FunSR)

🚀️🚀️🚀️ The repository will be orginized later.

## How to train a SR model

### Fixed scale SR model (*e.g.*, TransEnet, SRCNN, LGCNet, FSRCNN, DCM, VDSR)

Please run **"train_cnn_sr.py"**. Config is the file of **"configs/train_CNN.yaml**.

### Continuous scale SR model (*e.g.*, LIIF, MetaSR, ALIIF)

Please run **"train_liif_metasr_aliff.py"**. Config are in the folder of **"configs/baselines/"**.

### Continuous scale SR model (*e.g.*, DIINN, ArbRCAN, SADN, OverNet)

Please run **"train_diinn_arbrcan_sadn_overnet.py"**. Config is the file of **"configs/baselines/train_1x-5x_INR_diinn_arbrcan_sadn_overnet.yaml"**.

### **Our FunSR model**

Please run **"train_inr_funsr.py"** for a single GPU training or DP multi GPUs training. Config is the file of **"configs/train_1x-5x_INR_funsr.yaml"**.

👍👍👍 We also provide a DDP multi GPUs training method. Please refer to **"scripts/train_multi_gpus_with_ddp.py"** and **"train_inr_funsr_ddp.py"** for more details.

## How to eval a SR model

### Interpolation SR model (*e.g.*, Bilinear, Bicubic)

Please run **"test_interpolate_sr.py"**. Config is the file of **"configs/test_interpolate.yaml**.

### Fixed scale SR model (*e.g.*, TransEnet, SRCNN, LGCNet, FSRCNN, DCM, VDSR)

Please run **"test_cnn_sr.py"**. Config is the file of **"configs/test_CNN.yaml**.

### Continuous scale SR model (*e.g.*, LIIF, MetaSR, ALIIF)

Please run **"test_inr_liif_metasr_aliif.py"**. Config is the file of **"configs/baselines/test_INR_liif_metasr_aliif.yaml"**.

### Continuous scale SR model (*e.g.*, DIINN, ArbRCAN, SADN, OverNet, **FunSR**)

Please run **"test_inr_diinn_arbrcan_sadnarc_funsr_overnet.py"**. Config is the file of **"configs/test_INR_diinn_arbrcan_funsr_overnet.yaml"**.

👍👍👍 We also provide a script to evaluate multi upscale factors in a batch. Please refer to **"scripts/test_script.py"** for more details.

## Result visualization

We provide some visualization tools to show the reconstruction results in our paper. Please refer to the folder **"tools/paper_vis_tools"** for more details.

```
@misc{chen2023continuous,
      title={Continuous Remote Sensing Image Super-Resolution based on Context Interaction in Implicit Function Space}, 
      author={Keyan Chen and Wenyuan Li and Sen Lei and Jianqi Chen and Xiaolong Jiang and Zhengxia Zou and Zhenwei Shi},
      year={2023},
      eprint={2302.08046},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

If you have any questions, please feel free to reach me.


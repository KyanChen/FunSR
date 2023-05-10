# FunSR - Continuous Remote Sensing Image Super-Resolution based on Context Interaction in Implicit Function Space

English | [简体中文](/readme_cn.md)

This is the pytorch implement of our paper "Continuous Remote Sensing Image Super-Resolution based on Context Interaction in Implicit Function Space"


[Project Page](https://kyanchen.github.io/FunSR/) $\cdot$ [PDF Download](https://arxiv.org/abs/2302.08046) $\cdot$ [HuggingFace Demo](https://huggingface.co/spaces/KyanChen/FunSR)


## 0. Environment Setup

### 0.1 Create a virtual environment

```shell
conda create -n FunSR python=3.10
```

### 0.2 Activate the virtual environment
```sehll
conda activate FunSR
```

### 0.3 Install pytorch
Version of 1.x is also work, but the version of 2.x is recommended.
```shell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
```

### 0.3 [Optional] Install pytorch
```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

### 0.4 Install mmcv
Version of 2.x is recommended, but the version of 1.x is also work.
```shell
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html
```
Please refer to [installation documentation](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) for more detailed installation.

### 0.5 Install other dependencies
```shell
pip install -r requirements.txt
```

## 1. Data Preparation

### 1.1 Download the dataset
Put the downloaded HR images into the **samples** folder. In this project, some example images are provided in this folder.

### 1.2 Split the training and validation set
The data split files in the paper are provided in the **data_split** folder; if you need to split the training and validation set by yourself, please use **tools/data_tools/get_train_val_list.py** to split the training and validation set, run ```python tools/data_tools/get_train_val_list.py```.


## 2. Model Training

### 1.1 FunSR
#### 1.1.1 Config file
The config file of FunSR is **configs/train_1x-5x_INR_funsr.yaml**. You can modify the parameters in this file according to the situation.

#### 1.1.2 Training
Run ```python train_inr_funsr.py``` to train the FunSR model. And you can modify the ArgumentParser parameters in this file according to the situation.

### 1.2 [Optinal] Fixed-scale SR models (TransEnet, SRCNN, LGCNet, FSRCNN, DCM, VDSR)
#### 1.2.1 Config file
The config file of fixed-scale SR models is **configs/baselines/train_CNN.yaml**.
#### 1.2.2 Training
Run ```python train_cnn_sr.py``` to train the fixed-scale SR models.

### 1.3 [Optional] Continuous-scale SR models (LIIF, MetaSR, ALIIF)
#### 1.3.1 Config file
The config file of continuous-scale SR models is **configs/baselines/train_1x-5x_INR_[liif, metasr, aliif].yaml**.
#### 1.3.2 Training
Run ```python train_liif_metasr_aliff.py``` to train the continuous-scale SR models.

### 1.4 [Optional] Continuous-scale SR models (DIINN, ArbRCAN, SADN, OverNet)
#### 1.4.1 Config file
The config file of continuous-scale SR models is **configs/baselines/train_1x-5x_INR_diinn_arbrcan_sadn_overnet.yaml**.
#### 1.4.2 Training
Run ```python train_diinn_arbrcan_sadn_overnet.py``` to train the continuous-scale SR models.

## 3. Model Evaluation

### 3.1 FunSR (including DIINN, ArbRCAN, SADN, OverNet)
#### 3.1.1 Config file
The config file of FunSR is **configs/test_INR_diinn_arbrcan_funsr_overnet.yaml**. You can modify the parameters in this file according to the situation.
#### 3.1.2 Testing
Run ```python test_inr_diinn_arbrcan_sadnarc_funsr_overnet.py``` to test the FunSR model. And you can modify the ArgumentParser parameters in this file according to the situation.

### 3.2 [Optional] Fixed-scale SR models (TransEnet, SRCNN, LGCNet, FSRCNN, DCM, VDSR)
#### 3.2.1 Config file
The config file of fixed-scale SR models is **configs/baselines/test_CNN.yaml**.
#### 3.2.2 Testing
Run ```python test_cnn_sr.py``` to test the fixed-scale SR models.

### 3.3 [Optional] Continuous-scale SR models (LIIF, MetaSR, ALIIF)
#### 3.3.1 Config file
The config file of continuous-scale SR models is **configs/baselines/test_1x-5x_INR_[liif, metasr, aliif].yaml**.
#### 3.3.2 Testing
Run ```python test_liif_metasr_aliff.py``` to test the continuous-scale SR models.

### 3.4 [Optional] Continuous-scale SR models (DIINN, ArbRCAN, SADN, OverNet)
#### 3.4.1 Config file
The config file of continuous-scale SR models is **configs/baselines/test_1x-5x_INR_diinn_arbrcan_sadn_overnet.yaml**.
#### 3.4.2 Testing
Run ```python test_diinn_arbrcan_sadn_overnet.py``` to test the continuous-scale SR models.

### 3.5 [Optional] Interpolation-based SR models (Bicubic, Bilinear)
#### 3.5.1 Config file
The config file of interpolation-based SR models is **configs/test_interpolate.yaml**.
#### 3.5.2 Testing
Run ```python test_interpolate_sr.py``` to test the interpolation-based SR models.

## 4. [optional] Result Visualization
Some visualization tools are provided in the **tools/paper_vis_tools** folder, you can refer to the files in this folder for details.


## 5. [optional] Model Download
The model weights of RDN are provided in the [huggingface space](https://huggingface.co/spaces/KyanChen/FunSR/tree/main/pretrain).

## 6. [optional] Citation
If you find this project useful for your research, please cite our paper.
If you have any other questions, please contact me!!!
    
```
@article{chen2023continuous,
  title={Continuous Remote Sensing Image Super-Resolution based on Context Interaction in Implicit Function Space},
  author={Chen, Keyan and Li, Wenyuan and Lei, Sen and Chen, Jianqi and Jiang, Xiaolong and Zou, Zhengxia and Shi, Zhenwei},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2023},
  publisher={IEEE}
}
```
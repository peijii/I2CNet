# I2CNet
Code for methods in the paper: Intra- and Inter-Channel Deep Convolutional Neural Network with Dynamic Label Smoothing for Multichannel Biosignal Analysis
## Architecture of I2CNet with Dynamic Label Smoothing
![overall structure](fig/I2CNet.png)
>Detailed architecture of our proposed neural network. The proposed architecture mainly consists of a `feature extractor`, a `label predictor`, and `label adjustor` which not included in a standard feed-forward neural network. During training phase, both label predictor and label adjustor can supervise the feature extractor.
## Different components of I2CNET
1. ***I2C Convolutional Block***
<p align="center">
  <img src="fig/I2CBlock.png" alt="模块1" width="45%">
</p>

>Structure of two kinds of I2C convolution block. (a) First layer convolution block. (b) Non-first layer convolution block.

#### Code Implementation ####
* `I2CNet/src/models/I2CNet.py`
  * [class I2CBlockv1](https://github.com/peijii/I2CNet/blob/59eee8320e6a7c553711a067245e5b36b06deda7/src/models/I2CNet.py#L211): Implementation a I2C block of type (a).
  * [class I2CBlockv2](https://github.com/peijii/I2CNet/blob/59eee8320e6a7c553711a067245e5b36b06deda7/src/models/I2CNet.py#L270): Implementation a I2C block of type (b).

2. ***I2C MSE Module***
<p align="center">
  <img src="fig/I2CMSE.png" alt="模块2" width="30%">
</p>

>Structure of I2C MSE module.

#### Code Implementation ####
* `I2CNet/src/models/I2CNet.py`
  * [class I2CMSE](https://github.com/peijii/I2CNet/blob/59eee8320e6a7c553711a067245e5b36b06deda7/src/models/I2CNet.py#L341): Implementation of I2CMSE Module.

3. ***I2C Attention Module***
<p align="center">
  <img src="fig/I2CAttention.png" alt="模块2" width="28%">
</p>

>Structure of I2C Attention module.

#### Code Implementation ####
* `I2CNet/src/models/I2CNet.py`
  * [class I2CAttention](https://github.com/peijii/I2CNet/blob/59eee8320e6a7c553711a067245e5b36b06deda7/src/models/I2CNet.py#L539): Implementation of I2CAttention Module.
 
4. ***Dynamic Label Smoothing Module***
<p align="center">
  <img src="fig/DLS.png" alt="模块2" width="32%">
</p>

>Structure of DLS module.

#### Code Implementation ####
* `I2CNet/src/models/DLS.py`
  * [class DynamicLabelSmoothing](https://github.com/peijii/I2CNet/blob/59eee8320e6a7c553711a067245e5b36b06deda7/src/models/DLS.py#L52): Implementation of Dynamic Label Smoothing Module.

## Datasets
We evaluate our proposed method on the ISRUC-S3 dataset, the HEF dataset and the Ninapro-DB1.
* The **ISRUC-S3 dataset** is available [here](https://sleeptight.isr.uc.pt/), and we provide a detailed pipeline to run I2CNet and I2CNet + DLS on it.
* Tge **HEF dataset** is our previous research, and is available [here](https://github.com/peijii/a-layered-sensor-unit/tree/main/main_experiment/dataset).
* The **Ninapro-DB1** is available [here](https://ninapro.hevs.ch/instructions/DB1.html).

## Requirements

* Python 3.8
* Pytorch 1.11.0
* sklearn 0.24.0

## Usage 
- **1. Download the ISRUC-S3 Dataset:**
  
  Download the ISRUC-S3 dataset from the following command.

  ```shell
  ./download_ISRUC_S3.sh
  ```
  
- **2. Data preparation and processing:**
  
  To speed up the training process, save each sample as a separate .mat file.

  ```shell
  python preprocess.py
  ```
  
- **3. Configuration:**
  
  We provide config files used in our research without DLS strategy `/config/ISRUCnoDLS.config` and with DLS strategy `/config/ISRUCwithDLS.config`. 

  
- **4. Train and Evaluate I2CNet without DLS:**
  
  Run `python train_I2CNetnoDLS.py` with -c parameters.

  ```shell
  python train_I2CNetnoDLS.py -c ./config/ISRUCnoDLS.config
  ```
  
- **5. Train and Evaluate I2CNet with DLS:**
  
  Run `python train_I2CNetwithDLS.py` with -c parameters.

  ```shell
  python train_I2CNetwithDLS.py -c ./config/ISRUCwithDLS.config
  ```

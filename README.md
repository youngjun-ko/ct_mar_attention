# Rigid and Non-rigid Motion Artifacts Reduction in X-ray CT using Attention Module

This repository offers the data and code introduced in the following paper:

["Rigid and Non-rigid Motion Artifacts Reduction in X-ray CT using Attention Module"](https://doi.org/10.1016/j.media.2020.101883).

<p align="center"><img width="800" src="https://user-images.githubusercontent.com/58386956/98059090-9ab77800-1e89-11eb-852f-2b285c72af59.png"></p>


## Dependencies
To utilize our code, make sure you have installed [Python](https://www.python.org/) and [Tensorflow](https://www.tensorflow.org/) on your system.

In our experiments, python2.7 and tensorflow1.8 have been used.


## Preparation
To download our code, clone this repository as follow:
```
git clone https://github.com/youngjun-ko/ct_mar_attention
cd ct_mar_attention
```

Prepare your data in following form:

  ```
  Folder name: './data'
  File format: '.npy' (numpy array)
  Data format: 'NHWC'
  ```

* **Our dataset** and **pre-trained models** can be downloaded from [here](https://drive.google.com/drive/folders/19vV5JpegyFUSPuqhgEWV0OPMeGp50a-l?usp=sharing)

* Pre-trained VGG model can be downloaded from [here](https://github.com/machrisaa/tensorflow-vgg)   

<br>

**Please note that the code and model have been updated on 2021.06.10.**

## Usage
To train and test our network, run:
```
python train.py
python test.py
```   

## Citation
If you find our work useful, please cite us:
```
@article{ko2021rigid,
  title={Rigid and non-rigid motion artifact reduction in X-ray CT using attention module},
  author={Ko, Youngjun and Moon, Seunghyuk and Baek, Jongduk and Shim, Hyunjung},
  journal={Medical Image Analysis},
  volume={67},
  pages={101883},
  year={2021},
  publisher={Elsevier}
}
```

## Contact
E-mail: youngjun.ko@yonsei.ac.kr

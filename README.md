# Rigid and Non-rigid Motion Artifacts Reduction in X-ray CT using Attention Module

<img width="800" src="https://user-images.githubusercontent.com/58386956/98059090-9ab77800-1e89-11eb-852f-2b285c72af59.png">

This repository offers the data and code introduced in the following paper:

["Rigid and Non-rigid Motion Artifacts Reduction in X-ray CT using Attention Module"](https://doi.org/10.1016/j.media.2020.101883).

```
@article{ko2020rigid,
  title={Rigid and non-rigid motion artifact reduction in X-ray CT using attention module},
  author={Ko, Youngjun and Moon, Seunghyuk and Baek, Jongduk and Shim, Hyunjung},
  journal={Medical Image Analysis},
  pages={101883},
  year={2020},
  publisher={Elsevier}
}
```

## Dependencies
To utilize our code, you need to install followings on your system:
* [Python](https://www.python.org/)
* [Numpy](https://numpy.org/)
* [Tensorflow](https://www.tensorflow.org/) 

## Preparation
To download our code, clone this repository as follow:
```
git clone https://github.com/youngjun-ko/ct_mar_attention
cd ct_mar_attention
```

### **Download our Dataset and Pre-trained models**   
Our dataset and trained models can download [here](https://drive.google.com/drive/folders/1L0Mm8XM7_3oao3eXqNib03FZRYLceKjM?usp=sharing)   

### **Pre-trained VGG model**   
Pre-trained VGG model can be downloaded from [here](https://github.com/machrisaa/tensorflow-vgg)   

### **Prepare your data in following form:**   
* ```folder name```: './data'
* ```file format```: '.npy' (numpy array)
* ```data format```: 'NHWC'

## Usage
To train and test our network, run:
```
python train.py
python test.py
```   

## Contact
E-mail: youngjun.ko@yonsei.ac.kr

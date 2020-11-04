# Rigid and Non-rigid Motion Artifacts Reduction in X-ray CT using Attention Module
This repository offers the data and code introduced in the paper:
["Rigid and Non-rigid Motion Artifacts Reduction in X-ray CT using Attention Module"](https://doi.org/10.1016/j.media.2020.101883).

## Dependencies
To utilize our code, you need to install followings on your system:
* [Python](https://www.python.org/)
* [Numpy](https://numpy.org/)
* [Tensorflow](https://www.tensorflow.org/) 

## Usage
### **Download our code**   
To download our code, clone this repository as follow:
```
git clone https://github.com/youngjun-ko/ct_mar_attention
cd ct_mar_attention
```

### **Download our Dataset and Pre-trained models**   
Our dataset and trained models can download [here](https://drive.google.com/drive/folders/1L0Mm8XM7_3oao3eXqNib03FZRYLceKjM?usp=sharing)   

### **Pre-trained VGG model**   
Pre-trained VGG model can be downloaded from [here](https://github.com/machrisaa/tensorflow-vgg)   

### **Training & Testing network**   
```
python train.py
python test.py
```   

### **Prepare your data in following form:**   
* ```folder name```: './data'
* ```file format```: '.npy' (numpy array)
* ```data format```: 'NHWC'


## Contact
E-mail: youngjun.ko@yonsei.ac.kr

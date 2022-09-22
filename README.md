### Dependencies
* Python 3.7.6
* PyTorch 1.7.0 
* torchvision 0.8.0 

### custom_I3D : classfication fine-tuning for I3D
```
$ git clone https://github.com/Dahee9532/capstone.git
$ cd capstone/custom_I3D
$ python train.py
```

### custom_X3D : classfication fine-tuning for X3D
``` 
$ git clone https://github.com/Dahee9532/capstone.git
$ cd capstone/custom_X3D
$ python train.py
```


### 3D_CNN_nets ( C3D, P3D, I3D, X3D, SlowFast )
``` 
$ git clone https://github.com/Dahee9532/capstone.git
$ cd capstone/3D_CNN_nets
```

> #### /capstone/3D_CNN_nets/main.py
* path : data input path
* target_dir : numpy output path
* net : 3D CNN model

<img width="646" alt="스크린샷 2022-09-22 오후 2 20 00" src="https://user-images.githubusercontent.com/107235450/191664465-e08f172b-f3a2-4150-b338-991fe14de988.png">

```
$ python train.py
```


### Reference
* C3D : https://github.com/jfzhang95/pytorch-video-recognition
* P3D : https://github.com/qijiezhao/pseudo-3d-pytorch
* I3D : https://github.com/piergiaj/pytorch-i3d
* X3D : https://github.com/kkahatapitiya/X3D-Multigrid
* SlowFast : https://github.com/facebookresearch/SlowFast

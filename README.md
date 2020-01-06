# CrDoCo: Pixel-level Domain Transfer with Cross-Domain Consistency

This repository contains the code for the paper CrDoCo: Pixel-level Domain Transfer with Cross-Domain Consistency.

<img src="img/teaser.png" width="1000">

## Abstract
Unsupervised domain adaptation algorithms aim to transfer the knowledge learned from one domain to another (e.g., synthetic to real images). The adapted representations often do not capture pixel-level domain shifts that are crucial for dense prediction tasks (e.g., semantic segmentation). In this paper, we present a novel pixel-wise adversarial domain adaptation algorithm. By leveraging image-to-image translation methods for data augmentation, our key insight is that while the translated images between domains may differ in styles, their predictions for the task should be consistent. We exploit this property and introduce a cross-domain consistency loss that enforces our adapted model to produce consistent predictions. Through extensive experimental results, we show that our method compares favorably against the state-of-the-art on a wide variety of unsupervised domain adaptation tasks.

## Citation
If you find our code useful, please consider citing our work using the following bibtex:
```
@inproceedings{CrDoCo,
  title={CrDoCo: Pixel-level Domain Transfer with Cross-Domain Consistency},
  author={Chen, Yun-Chun and Lin, Yen-Yu and Yang, Ming-Hsuan and Huang, Jia-Bin },
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```

## Environment
 - Install Anaconda Python3.7
 - This code is tested on NVIDIA RTX 2080 GPU with 24GB memory
 

## Dataset
 - Please download the [Cityscapes](https://www.cityscapes-dataset.com), [GTA 5](https://download.visinf.tu-darmstadt.de/data/from_games/), [NYUv2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html), and [SUNCG](http://suncg.cs.princeton.edu/) datasets.


## Demo code
 
We prepare a demo code so that you can have a better understanding on the workflow of the code. Please refer to `demo.py`


## Pre-training the image-to-image translation network
 
We use the source code from [Zhu et al.](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). Please follow the training tips [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md) for pre-training the image-to-image translation networks. Please adjust the image size based on the GPU memory.

### Note: We found that the training of image-to-image translation network is not very stable. Sometimes, the results will have severe visual artifacts or distortions. Based on our experience, we found that resizing images to resolution higher than 384x384 will implicitly alleviate this issue.


## Training the task network

``` 
python train.py --model train --img_source_file /path/to/source/dataset --img_target_file /path/to/target/dataset --lab_source_file /path/to/source/label --lab_target_file /path/to/target/label --shuffle --flip --rotation
```


## Evaluation
 
``` 
python test.py --model test --img_source_file /path/to/source/dataset --img_target_file /path/to/target/dataset
```

## Acknowledgement
 - This code is heavily borrowed from [Zheng et al.](https://github.com/lyndonzheng/Synthetic2Realistic), [Tsai et al.](https://github.com/wasidennis/AdaptSegNet), [Zhu et al.](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), [Hoffman et al.](https://github.com/jhoffman/cycada_release)

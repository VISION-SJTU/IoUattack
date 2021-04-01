# IoUattack

:herb: **[IoU Attack: Towards Temporally Coherent Black-Box Adversarial Attack for Visual Object Tracking](https://arxiv.org/pdf/2103.14938.pdf)**

Shuai Jia, Yibing Song, Chao Ma and Xiaokang Yang

*IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.*

## Introduction

<img src="https://github.com/VISION-SJTU/IoUattack/blob/main/demo/intro.png" width='500'/><br/>

We observe that the increase of noise level positively correlates to the decrease of IoU scores, but their directions are not exactly the same.
- Our IoU attack seeks to inject the lowest amount of noisy perturbations at the same contour line of IoU score for each iteration.
- We choose three representative trackers with different structures, **SiamRPN++**, **DiMP** and **LTMU**, respectively.


## Code

- The code and raw results will be released soon!! :star: :star: :star:

## Demo

<img src="https://github.com/VISION-SJTU/IoUattack/blob/main/demo/car_clean.gif" width='300'/>   <img src="https://github.com/VISION-SJTU/IoUattack/blob/main/demo/car_attack.gif" width='300'/><br/>
&emsp; &emsp;&emsp;&emsp;&emsp;&emsp; <img src="https://github.com/VISION-SJTU/IoUattack/blob/main/demo/legend.png" width='300'/><br/>


## Citation
If any part of our paper and code is helpful to your work, please generously citing: 
```
@inproceedings{jia-cvpr21-iouattack,
  title={IoU Attack: Towards Temporally Coherent Black-Box Adversarial Attack for Visual Object Tracking},
  author={Jia, Shuai and Song, Yibing and Ma, Chao and Yang, Xiaokang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```

Thank you :)

## License
Licensed under an MIT license.

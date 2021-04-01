# IoUattack

:herb: **[IoU Attack: Towards Temporally Coherent Black-Box Adversarial Attack for Visual Object Tracking](https://arxiv.org/pdf/2103.14938.pdf)**

Shuai Jia, Yibing Song, Chao Ma and Xiaokang Yang

*IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.*

## Introduction

<img src="https://github.com/VISION-SJTU/IoUattack/blob/main/demo/intro.png" width='500'/><br/>

We observe that the increase of noise level positively correlates to the decrease of IoU scores, but their directions are not exactly the same.
- Our IoU attack seeks to inject the lowest amount of noisy perturbations at the same contour line of IoU score for each iteration.
- We choose three representative trackers with different structures, **SiamRPN++**, **DiMP** and **LTMU**, respectively.

## Results

 #### Result for SiamRPN++ on multiple datasets
|                   | VOT2019<br>A / R / EAO  |  VOT2018<br>A / R / EAO  | VOT2016<br>A / R / EAO | VOT2018lt <br> F-score| OTB2015<br>OP / DP| NFS30<br>OP / DP| 
| ------------------| :--------------------:  | :----:  |:----: |:----: |:----: |:----: |
| SiamRPN++         | 0.596 / 0.472 / 0.287   |  0.602 / 0.239 / 0.413   |0.643 / 0.200 / 0.461|  0.625 | 0.695 / 0.905    | 0.509 / 0.601    |
| SiamRPN++(Random) | 0.591 / 0.727 / 0.220   |  0.587 / 0.365 / 0.301   |0.632 / 0.340 / 0.331|  0.553 | 0.631 / 0.818    | 0.466 / 0.550    |
| SiamRPN++(Attack) | 0.575 / 1.575 / 0.124   |  0.568 / 1.171 / 0.129   |0.605 / 0.802 / 0.183|  0.453 | 0.499 / 0.644    | 0.394 / 0.446    |


 #### Result for DiMP on multiple datasets
|                   | VOT2019<br>A / R / EAO  |  VOT2018<br>A / R / EAO  | VOT2016<br>A / R / EAO | VOT2018lt <br> F-score| OTB2015<br>OP / DP| NFS30<br>OP / DP| 
| ------------------| :--------------------:  | :----:  |:----: |:----: |:----: |:----: |
| DiMP              | 0.568 / 0.277 / 0.332   |  0.574 / 0.145 / 0.427   |0.599 / 0.140 / 0.449|  0.609 | 0.671 / 0.869    | 0.614 / 0.729    |
| DiMP(Random)      | 0.567 / 0.373 / 0.284   |  0.560 / 0.202 / 0.363   |0.592 / 0.168 / 0.404|  0.555 | 0.659 / 0.860    | 0.591 / 0.710    |
| DiMP(Attack)      | 0.474 / 1.073 / 0.195   |  0.507 / 0.400 / 0.248   |0.536 / 0.374 / 0.256|  0.443 | 0.592 / 0.791    | 0.545 / 0.658    |

 #### Result for LTMU on multiple datasets
|                   | VOT2019<br>A / R / EAO  |  VOT2018<br>A / R / EAO  | VOT2016<br>A / R / EAO | VOT2018ltT <br> F-score| OTB2015<br>OP / DP| NFS30<br>OP / DP| 
| ------------------| :--------------------:  | :----:  |:----: |:----: |:----: |:----: |
| LTMU              | 0.625 / 0.913 / 0.201   |  0.624 / 0.702 / 0.195   |0.661 / 0.522 / 0.236|  0.691 | 0.672 / 0.872    | 0.631 / 0.764    |
| LTMU(Random)      | 0.623 / 1.073 / 0.175   |  0.622 / 0.805 / 0.178   |0.646 / 0.592 / 0.233|  0.657 | 0.622 / 0.815    | 0.579 / 0.699    |
| LTMU(Attack)      | 0.576 / 1.470 / 0.150   |  0.590 / 1.320 / 0.120   |0.604 / 0.904 / 0.170|  0.589 | 0.517 / 0.712    | 0.462 / 0.559    |

:herb: **All raw results are available.**  [[Google_drive]](https://drive.google.com/drive/folders/1WjYJzsLEJZkB1dw-17ZLJNYZ9THK-jL4?usp=sharing)  [[Baidu_Disk]](https://pan.baidu.com/s/1HD5LEQfWvC0bV7xxW_jY-A) Code: c7ew


## Code

- The code will be released soon!! :star: :star: :star:

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

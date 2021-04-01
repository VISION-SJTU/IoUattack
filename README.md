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
|                   | VOT2019<br>A / R / EAO  |  VOT2018<br>A / R / EAO  | VOT2016<br>A / R / EAO | VOT2018-LT <br> F-score| OTB2015<br>OP / DP| NFS30<br>OP / DP| 
| ------------------| :--------------------:  | :----:  |:----: |:----: |:----: |:----: |
| SiamRPN++         | 0.596 / 0.472 / 0.287   |   0.585 / 0.272 / 0.380  |0.622 / 0.214 / 0.418|  0.592 | 0.592 / 0.791    | 0.592 / 0.791    |
| SiamRPN++(Random) | 0.591 / 0.727 / 0.220   |  0.571 / 0.529 / 0.223   |0.606 / 0.303 / 0.336|  0.592 | 0.592 / 0.791    | 0.592 / 0.791    |
| SiamRPN++(Attack) | 0.575 / 1.575 / 0.124   |  0.536 / 1.447 / 0.097   |0.521 / 1.631 / 0.078|  0.592 | 0.592 / 0.791    | 0.592 / 0.791    |


 #### Result for DiMP on multiple datasets
|                   | VOT2019<br>A / R / EAO  |  VOT2018<br>A / R / EAO  | VOT2016<br>A / R / EAO | VOT2018-LT <br> F-score| OTB2015<br>OP / DP| NFS30<br>OP / DP| 
| ------------------| :--------------------:  | :----:  |:----: |:----: |:----: |:----: |
| DiMP              | 0.585 / 0.272 / 0.380   |   0.585 / 0.272 / 0.380  |0.622 / 0.214 / 0.418|  0.592 | 0.592 / 0.791    | 0.592 / 0.791    |
| DiMP(Random)      | 0.585 / 0.272 / 0.380   |  0.571 / 0.529 / 0.223   |0.606 / 0.303 / 0.336|  0.592 | 0.592 / 0.791    | 0.592 / 0.791    |
| DiMP(Attack)      | 0.585 / 0.272 / 0.380   |  0.536 / 1.447 / 0.097   |0.521 / 1.631 / 0.078|  0.592 | 0.592 / 0.791    | 0.592 / 0.791    |

 #### Result for LTMU on multiple datasets
|                   | VOT2019<br>A / R / EAO  |  VOT2018<br>A / R / EAO  | VOT2016<br>A / R / EAO | VOT2018-LT <br> F-score| OTB2015<br>OP / DP| NFS30<br>OP / DP| 
| ------------------| :--------------------:  | :----:  |:----: |:----: |:----: |:----: |
| LTMU              | 0.585 / 0.272 / 0.380   |   0.585 / 0.272 / 0.380  |0.622 / 0.214 / 0.418|  0.592 | 0.592 / 0.791    | 0.592 / 0.791    |
| LTMU(Random)      | 0.585 / 0.272 / 0.380   |  0.571 / 0.529 / 0.223   |0.606 / 0.303 / 0.336|  0.592 | 0.592 / 0.791    | 0.592 / 0.791    |
| LTMU(Attack)      | 0.585 / 0.272 / 0.380   |  0.536 / 1.447 / 0.097   |0.521 / 1.631 / 0.078|  0.592 | 0.592 / 0.791    | 0.592 / 0.791    |

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

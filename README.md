# Towards Distribution-Agnostic Generalized Category Discovery.

**NIPS 2023:** This repository is the official implementation of [BaCon]().

## Introduction
Data imbalance and open-ended distribution are two intrinsic characteristics of the real visual world. Though encouraging progress has been made in tackling each challenge separately, few works dedicated to combining them towards real-world scenarios. While several previous works have focused on classifying close-set samples and detecting open-set samples during testing, it's still essential to be able to classify unknown subjects as human beings. In this paper, we formally define a more realistic task as distribution-agnostic generalized category discovery (DA-GCD): generating fine-grained predictions for both close- and open-set classes in a long-tailed open-world setting. To tackle the challenging problem, we propose a Self-**Ba**lanced **Co**-Advice co**n**trastive framework (BaCon), which consists of a contrastive-learning branch and a pseudo-labeling branch, working collaboratively to provide interactive supervision to resolve the DA-GCD task. In particular, the contrastive-learning branch provides reliable distribution estimation to regularize the predictions of the pseudo-labeling branch, which in turn guides contrastive learning through self-balanced knowledge transfer and a proposed novel contrastive loss. We compare BaCon with state-of-the-art methods from two closely related fields: imbalanced semi-supervised learning and generalized category discovery. The effectiveness of BaCon is demonstrated with superior performance over all baselines and comprehensive analysis across various datasets.

## Method
<div align=center>
<img src="pipeline.png" width="800" >
</div>

Overview of the self-balanced co-advice contrastive framework (BaCon).

## Environment
Requirements:
```
loguru
numpy
pandas
scikit_learn
scipy
torch==1.10.0
torchvision==0.11.1
tqdm
```

## Data
We provide the specific train split of CIFAR-10 and CIFAR-100 with different imbalance ratios, please refer to ```data_uq_idxs``` for details. We also provide the source code in ```data/imagenet.py``` for splitting data to the proposed DA-GCD setting.


## Pretrained models downloading
[CIFAR-10](https://drive.google.com/file/d/1OFcLeDK1HUD6N9pQuRHp_TJ0CaTybWxJ/view?usp=sharing)

[CIFAR-100](https://drive.google.com/file/d/1pbZukDXHqwUtvfP24lqlvKT5V2UHfOGk/view?usp=sharing)

[ImageNet-100](https://drive.google.com/file/d/1L_sL4B7WhyPeqna5hYSpP2k52FKbDxQl/view?usp=sharing)

## Training
### CIFAR-10
```
bash run_cifar10.sh
```

### CIFAR-100
```
bash run_cifar100.sh
```

### ImageNet-100
```
bash run_imagenet100.sh
```

## Acknowledgments
The codebase is largely built on [GCD](https://github.com/CVMI-Lab/SimGCD) and [SimGCD](https://github.com/CVMI-Lab/SimGCD). Thanks for their great work!

## Citation
```
@misc{bai2023distributionagnostic,
      title={Towards Distribution-Agnostic Generalized Category Discovery}, 
      author={Jianhong Bai and Zuozhu Liu and Hualiang Wang and Ruizhe Chen and Lianrui Mu and Xiaomeng Li and Joey Tianyi Zhou and Yang Feng and Jian Wu and Haoji Hu},
      year={2023},
      eprint={2310.01376},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## What's More?
Our work on self-supervised long-tail learning: [On the Effectiveness of Out-of-Distribution Data on Self-Supervised Long-Tail Learning.](https://arxiv.org/abs/2306.04934v2)

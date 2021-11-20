# ReSSL: Relational Self-Supervised Learning with Weak Augmentation (NeurIPS 2021)

This repository contains PyTorch evaluation code, training code and pretrained models for ReSSL.

For details see [ReSSL: Relational Self-Supervised Learning with Weak Augmentation](https://proceedings.neurips.cc/paper/2021/file/14c4f36143b4b09cbc320d7c95a50ee7-Paper.pdf) by Mingkai Zheng, Shan You, Fei Wang, Chen Qian, Changshui Zhang, Xiaogang Wang and Chang Xu

![ReSSL](img/framework.png)


## Reproducing

To run the code, you probably need to change the Dataset setting (dataset/imagenet.py), and Pytorch DDP setting (util/dist_init.py) for your own server enviroments.

The distribued training of this code is base on slurm enviroments, we have provide the training scrips under the script folder.


We also provide the pretrained model for ResNet50 (single crop and 5 crops)

|          |Arch | BatchSize | Epochs | Crops | Linear Eval | Download  |
|----------|:----:|:---:|:---:|:---:|:---:|:---:|
|  ReSSL | ResNet50 | 256 | 200  | 1 | 69.9 % | [ressl-200.pth](https://drive.google.com/file/d/16Ib4rvEvB_rdQThPxkoOb9wvCALzPTZd/view?usp=sharing) |
|  ReSSL | ResNet50 | 256 | 200  | 5 | 74.7 % | [ressl-multi-200.pth](https://drive.google.com/file/d/1usvvFAw_1bOaiXBgxXG9kwOOPb0VAy0Y/view?usp=sharing) |

If you want to test the pretained model, please download the weights from the link above, and move it to the checkpoints folder (create one if you don't have .checkpoints/ directory). The evaluation scripts also has been provided in script/train.sh


## Citation
If you find that ReSSL interesting and help your research, please consider citing it:
```

@inproceedings{
      zheng2021ressl,
      title={Re{SSL}: Relational Self-Supervised Learning with Weak Augmentation},
      author={Mingkai Zheng and Shan You and Fei Wang and Chen Qian and Changshui Zhang and Xiaogang Wang and Chang Xu},
      booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
      year={2021},
      url={https://openreview.net/forum?id=ErivP29kYnx}
}
```


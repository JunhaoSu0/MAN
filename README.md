# Momentum Auxiliary Network for Supervised Local Learning (ECCV2024 Oral)
![image](https://github.com/JunhaoSu0/MAN/assets/174414200/deff0518-f88f-45f9-a743-8861067fb62a)
**This Figure shows our overall arch.**


![image](https://github.com/JunhaoSu0/MAN/assets/174414200/e99468a4-70d5-44b5-b565-75461e28d7ba)

**This Figure shows detail of our MAN.**

# To Train on Different Datasets

**For CIFAR-10/STL-10/SVHN:**
```
cd Exp\ on\ CIFAR/SVHN/STL
```
**For eaxmple: ResNet-32 (K=16) on CIFAR-10:**
```
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --no 0 --cos_lr --local_module_num 16  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128 --ixx_1 5 --ixy_1 0.5 --ixx_2 0   --ixy_2 0  --momentum 0.995
```

**For ImageNet:**
```
cd Exp\ on\ ImageNet
```
**For eaxmple: ResNet-152 (K=2) on ImageNet:**
```
CUDA_VISIBLE_DEVICES=0 python imagenet_DDP.py  ./data/imagenet1K --arch resnetInfoPro_MAN --net resnet152 --local_module_num 2 --batch-size 128 --lr 0.05 --epochs 90 --workers 24 --gpu 0 --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --ixx_r 5 --ixy_r 0.75 --momentum_MAN 0.995
```
# Citation
```
@article{su2024momentum,
  title={Momentum Auxiliary Network for Supervised Local Learning},
  author={Su, Junhao and Cai, Changpeng and Zhu, Feiyu and He, Chenghao and Xu, Xiaojie and Guan, Dongzhi and Si, Chenyang},
  journal={arXiv preprint arXiv:2407.05623},
  year={2024}
}
```


# Momentum Auxiliary Network for Supervised Local Learning (ECCV2024)
![image](https://github.com/JunhaoSu0/MAN/assets/174414200/deff0518-f88f-45f9-a743-8861067fb62a)
**This Figure shows our overall arch.**


![image](https://github.com/JunhaoSu0/MAN/assets/174414200/e99468a4-70d5-44b5-b565-75461e28d7ba)

**This Figure shows detail of our MAN.**

# To Train on Different Datasets

**For CIFAR-10/STL-10/SVHN:**
```
cd Exp\ on\ CIFAR/SVHN/STL
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --no 0 --cos_lr --local_module_num 16  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128 --ixx_1 5 --ixy_1 0.5 --ixx_2 0   --ixy_2 0  --momentum 0.995
```



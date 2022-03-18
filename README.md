# A PyTorch implementation for Residual Attention Networks 

This repository contains a [PyTorch](http://pytorch.org/) implementation of the paper [Residual Attention Network for Image Classification](https://arxiv.org/abs/1704.06904). The code is based on [ResidualAttentionNetwork-pytorch](https://github.com/tengshaofeng/ResidualAttentionNetwork-pytorch).

The original repository contains the following implementations:
* ResidualAttentionModel92U for training on the CIFAR10 dataset
* ResidualAttentionModel92 for training on the ImageNet dataset
* ResidualAttentionModel448 for images with larger resolution

## Train and test your model
Example usage for training on CIFAR10:
```sh
$ python main.py --name ResNet-92-32U --tensorboard
```
or
```sh
$ python main_mixup.py --name ResNet-92-32U --tensorboard
```
Example usage for testing your trained model (be sure to use the same network model):
```sh
$ python main.py --test ResNet-92-32U
```

Note: To switch the model you're training on, be sure to replace the imported model:
```python
from model.residual_attention_network import ResidualAttentionModel92U as ResidualAttentionModel
```
either on [main.py](https://github.com/Necas209/RAN-PyTorch/main.py) or [main_mixup.py](https://github.com/Necas209/RAN-PyTorch/main_mixup.py), depending if you use mixup or not.

### Tracking training progress with TensorBoard
To track training progress, this implementation uses [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard) which offers great ways to track and compare multiple experiments. To track PyTorch experiments in TensorBoard we use [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger) which can be installed with 
```shell
pip install tensorboard_logger
```

### Dependencies
* [PyTorch](http://pytorch.org/)

optional:
* [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)

## Results (from source repository)
| Model                                            | Dataset | Top-1 error |
|--------------------------------------------------|---------|-------------|
| RAN92U                                           | CIFAR10 | 4.6         |
| RAN92U (with mixup)                              | CIFAR10 | 3.35        |
| RAN92U (with mixup and simpler attention module) | CIFAR10 | 3.16        |

## Cite
If you use DenseNets in your work, please cite the original paper as:
```bibtex
@misc{1704.06904,
    Author = {Fei Wang and Mengqing Jiang and Chen Qian and Shuo Yang and Cheng Li and Honggang Zhang and Xiaogang Wang and Xiaoou Tang},
    Title = {Residual Attention Network for Image Classification},
    Year = {2017},
    Eprint = {arXiv:1704.06904},
}
```
If this implementation is useful to you and your project, also consider citing or acknowledging this code repository.
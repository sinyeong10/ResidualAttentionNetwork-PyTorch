import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.backends import cudnn
from torch.utils.data import Dataset
from torchsummary import summary
from torchvision import datasets

from tensorboard_logger import configure
from argparse import ArgumentParser
from train import *
from test import *

parser = ArgumentParser(description='PyTorch Residual Attention Network')
parser.add_argument('--epochs', default=300, type=int,
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=100, type=int,
                    help='total number of layers (default: 100)')
parser.add_argument('--growth', default=12, type=int,
                    help='number of new channels per layer (default: 12)')
parser.add_argument('--test', default='', type=str,
                    help='path to trained model (default: none)')
parser.add_argument('--name', default='DenseNet_BC_100_12', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')

best_prec1 = 0.
args = Namespace


def main() -> None:
    global args, best_prec1
    args = parser.parse_args()
    if args.tensorboard:
        configure(f"runs/{args.name}")

    print(args)

    # 디바이스 설정 (GPU 사용 가능하면 GPU, 아니면 CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # CIFAR-10 데이터셋에 대한 전처리 설정
    # transform = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),  # 무작위로 이미지 좌우 반전
    #     transforms.RandomCrop(32, padding=4),  # 무작위로 이미지 자르기 (32x32 크기로 자르고 패딩은 4로 설정)
    #     transforms.ToTensor(),  # 이미지를 텐서로 변환
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 이미지를 정규화
    # ])


    # Image Preprocessing
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        normalize
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # # CIFAR-10 데이터셋 로드
    # train_dataset = CIFAR10(root='../data', train=True, download=True, transform=transform)
    # test_dataset = CIFAR10(root='../data', train=False, download=False, transform=test_transform)

    # # 데이터 로더 정의
    # train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    # test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)


    # CIFAR-10 Dataset
    dataset = datasets.CIFAR10(root='../data/', train=True,
                                transform=transform, download=True)
    torch.manual_seed(13)
    val_size = 5000
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    test_dataset = datasets.CIFAR10(root='../data/', train=False,
                                    transform=test_transform)

    # Data Loader (Input Pipeline)
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                                shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                shuffle=True, **kwargs)

    # 사전 훈련된 ResNet 모델 불러오기
    model = resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)  # CIFAR-10 분류를 위해 마지막의 fully connected layer 수정
    model = model.to(device)

    # 손실 함수와 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 훈련 루프
    num_epochs = 300
    model.train()

    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        correct = 0
        total = 0

        print(len(train_loader))
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 100 == 99:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
                    f"Loss: {running_loss / 100:.4f}, Accuracy: {100 * correct / total:.2f}%")
                running_loss = 0.0

    # 테스트 루프
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # 테스트셋의 정확도 출력
    print(f"테스트 정확도: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    main()


#Epoch 40에 75%
#Epoch 90에 80%
#625
# 2024-11-16 15:41:25.801801: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# 2024-11-16 15:41:26.358539: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# Epoch [106/300], Batch [100/5625], Loss: 0.6175, Accuracy: 79.12%
# Epoch [106/300], Batch [200/5625], Loss: 0.5421, Accuracy: 80.62%
# Epoch [106/300], Batch [300/5625], Loss: 0.5375, Accuracy: 81.12%
# Epoch [106/300], Batch [400/5625], Loss: 0.5553, Accuracy: 81.16%
# Epoch [106/300], Batch [500/5625], Loss: 0.6195, Accuracy: 80.92%
# Epoch [106/300], Batch [600/5625], Loss: 0.5490, Accuracy: 81.19%
# Epoch [106/300], Batch [700/5625], Loss: 0.5061, Accuracy: 81.52%
# Epoch [106/300], Batch [800/5625], Loss: 0.6333, Accuracy: 81.20%
# Epoch [106/300], Batch [900/5625], Loss: 0.5958, Accuracy: 81.03%
# Epoch [106/300], Batch [1000/5625], Loss: 0.6375, Accuracy: 80.90%
# Epoch [106/300], Batch [1100/5625], Loss: 0.5938, Accuracy: 80.84%
# Epoch [106/300], Batch [1200/5625], Loss: 0.5295, Accuracy: 80.85%
# Epoch [106/300], Batch [1300/5625], Loss: 0.5608, Accuracy: 81.00%
# Epoch [106/300], Batch [1400/5625], Loss: 0.5951, Accuracy: 80.90%
# Epoch [106/300], Batch [1500/5625], Loss: 0.5674, Accuracy: 80.84%
# Epoch [106/300], Batch [1600/5625], Loss: 0.5763, Accuracy: 80.75%
# Epoch [106/300], Batch [1700/5625], Loss: 0.5481, Accuracy: 80.84%
# Epoch [106/300], Batch [1800/5625], Loss: 0.5659, Accuracy: 80.88%
# Epoch [106/300], Batch [1900/5625], Loss: 0.5356, Accuracy: 80.88%
# Epoch [106/300], Batch [2000/5625], Loss: 0.6162, Accuracy: 80.84%
# Epoch [106/300], Batch [2100/5625], Loss: 0.5597, Accuracy: 80.96%
# Epoch [106/300], Batch [2200/5625], Loss: 0.5250, Accuracy: 81.01%
# Epoch [106/300], Batch [2300/5625], Loss: 0.4864, Accuracy: 81.05%
# Epoch [106/300], Batch [2400/5625], Loss: 0.5349, Accuracy: 81.09%
# Epoch [106/300], Batch [2500/5625], Loss: 0.5725, Accuracy: 81.14%
# Epoch [106/300], Batch [2600/5625], Loss: 0.5695, Accuracy: 81.11%
# Epoch [106/300], Batch [2700/5625], Loss: 0.5480, Accuracy: 81.15%
# Epoch [106/300], Batch [2800/5625], Loss: 0.5580, Accuracy: 81.20%
# Epoch [106/300], Batch [2900/5625], Loss: 0.5540, Accuracy: 81.22%
# Epoch [106/300], Batch [3000/5625], Loss: 0.4815, Accuracy: 81.29%
# Epoch [106/300], Batch [3100/5625], Loss: 0.5930, Accuracy: 81.25%
# Epoch [106/300], Batch [3200/5625], Loss: 0.5180, Accuracy: 81.29%
# Epoch [106/300], Batch [3300/5625], Loss: 0.5895, Accuracy: 81.25%
# Epoch [106/300], Batch [3400/5625], Loss: 0.6001, Accuracy: 81.23%
# Epoch [106/300], Batch [3500/5625], Loss: 0.5630, Accuracy: 81.25%
# Epoch [106/300], Batch [3600/5625], Loss: 0.5284, Accuracy: 81.28%
# Epoch [106/300], Batch [3700/5625], Loss: 0.5638, Accuracy: 81.25%
# Epoch [106/300], Batch [3800/5625], Loss: 0.5931, Accuracy: 81.20%
# Epoch [106/300], Batch [3900/5625], Loss: 0.5752, Accuracy: 81.23%
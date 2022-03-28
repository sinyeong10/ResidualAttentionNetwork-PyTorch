from torch.backends import cudnn
from torch.utils.data import Dataset
from torchvision import transforms, datasets

from tensorboard_logger import configure
from argparse import ArgumentParser
from train import *
from test import *

parser = ArgumentParser(description='PyTorch Residual Attention Network')
parser.add_argument('--epochs', default=300, type=int,
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
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

    # Create model
    model = ResidualAttentionModel()
    model = model.cuda()

    # get the number of model parameters
    print(f'Number of model parameters: {sum([p.data.nelement() for p in model.parameters()])}')

    cudnn.benchmark = True

    if args.test:
        args.test = f"runs/{args.test}/checkpoint.pth.tar"
        if os.path.isfile(args.test):
            print(f"=> loading model '{args.test}'")
            checkpoint = torch.load(args.test)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print(f"=> loaded model '{args.test}'")
            test(model, test_loader)
        else:
            print(f"=> no model found at '{args.test}'")
        return

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        # Decaying Learning Rate
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(model, train_loader, criterion, optimizer, epoch, args)

        # evaluate on validation set
        prec1 = validate(model, val_loader, criterion, epoch, args)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, args=args)
    print('Best accuracy: ', best_prec1)

    test(model, test_loader)


if __name__ == "__main__":
    main()

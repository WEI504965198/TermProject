'''Train CIFAR10 with PyTorch.'''
import csv
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time

from PIL import Image
from matplotlib import pyplot as plt

from models.preact_resnet_CELU import preactresnet18
#from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args(args=[])

device = 'cuda:7' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

print('<<< Data Information >>>')
print('Train data :', len(trainset))
print('Test data :', len(testset), '\n')

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
class_num = len(classes)

lr_decay_epochs = [60, 80, 100]
lr_decay_rate = 0.1
exp_model_detail = 'Preact_ResNet_18_CELU'

# Model
print('==> Building model ' + exp_model_detail)
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
net = preactresnet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                                momentum=0.9,
                                weight_decay=5e-4,
                                nesterov=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

def make_dir(dirs):
    try:
        if not os.path.exists(dirs):
            os.makedirs(dirs)
    except Exception as err:
        print("create_dirs error!")
        print(dirs)
        exit()

experiments = os.getcwd() + '/experiments/' + exp_model_detail + '/'
checkpoint_path = experiments + 'checkpoint/'
checkpoint_current = checkpoint_path + 'checkpoint.pth.tar'
checkpoint_best = checkpoint_path + 'checkpoint_best.pth.tar'
log_path_train = experiments + 'train_log.csv'
log_path_test = experiments + 'test_log.csv'

make_dir(checkpoint_path)

#if args.resume:
# Load checkpoint.
print('==> Resuming from checkpoint..')
if os.path.exists(checkpoint_current):
    print("=> loading checkpoint")
    checkpoint = torch.load(checkpoint_current)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] + 1
else:
    pass

def adjust_learning_rate(epoch, lr_decay_epochs, lr_decay_rate, learning_rate, optimizer):
    """Sets the learning rate to the initial LR decayed by 0.2 every steep step"""
    steps = np.sum(epoch > np.asarray(lr_decay_epochs))
    if steps > 0:
        new_lr = learning_rate * (lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()

    train_loss = 0
    correct = 0
    total = 0
    print('=========================train===============================')
    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)

        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        if batch_idx % 100 == 0:
            print('Epoch: {} | Batch_idx: {} |  Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
                  .format(epoch, batch_idx, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        train_acc = 100. * correct / total

    return train_loss / (batch_idx + 1), train_acc.item()


def test(epoch):
    global best_acc
    net.eval()

    test_loss = 0
    correct = 0
    total = 0
    print('=========================test===============================')
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        print('# TEST : Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
              .format(test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        test_acc = 100. * correct / total
        return test_loss / (batch_idx + 1), test_acc.item()


for epoch in range(start_epoch, 300):
    adjust_learning_rate(epoch, lr_decay_epochs, lr_decay_rate, args.lr, optimizer)

    train_loss, train_acc = train(epoch)
    # print(train_loss, train_acc)
    test_loss, test_acc = test(epoch)
    scheduler.step()

    # save train.csv
    if os.path.exists(log_path_train) == False:
        with open(log_path_train, 'w', newline='') as train_writer_csv:
            header_list = ['epoch', 'loss', 'acc']
            train_writer = csv.DictWriter(train_writer_csv, fieldnames=header_list)
            train_writer.writeheader()
    with open(log_path_train, 'a', newline='') as train_writer_csv:
        train_writer = csv.writer(train_writer_csv)
        train_writer.writerow([epoch, str(train_loss), str(train_acc)])

    # save test.csv
    if os.path.exists(log_path_test) == False:
        with open(log_path_test, 'w', newline='') as test_writer_csv:
            header_list = ['epoch', 'loss', 'acc']
            test_writer = csv.DictWriter(test_writer_csv, fieldnames=header_list)
            test_writer.writeheader()
    with open(log_path_test, 'a', newline='') as test_writer_csv:
        test_writer = csv.writer(test_writer_csv)
        test_writer.writerow([epoch, str(test_loss), str(test_acc)])

    state = {
        'net': net.state_dict(),
        'acc': test_acc,
        'epoch': epoch,
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, checkpoint_current)

    # Save checkpoint.
    if test_acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': test_acc,
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, checkpoint_best)
        best_acc = test_acc
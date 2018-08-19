from __future__ import print_function
import argparse
import shutil
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.autograd import Function
from torch.utils import data
from termcolor import cprint

from modules.CNNRNN_Char import CNNRNN_Char
from utils.data_loader import Dataset
# Training settings

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/home/cvpr19/scottreed/DATA/CUB', help='data directory.')
parser.add_argument('--nclass', type=int, default=200, help='number of classes')
parser.add_argument('--doc_length', type=int, default=201, help='document length')
parser.add_argument('--image_dim', type=int, default=1024, help='image feature dimension')
parser.add_argument('--batch_size',type=int, default=40, help='number of sequences to train on in parallel')
parser.add_argument('--randomize_pair', type=int, default=0, help='if 1, images and captions of the same class are randomly paired.')
#parser.add_argument('--ids_file', type=str, default='trainids.txt', help='file specifying which class labels are used for training. Can also be trainvalids.txt')
parser.add_argument('--ids_file', type=str, default='trainvalids.txt', help='file specifying which class labels are used for training. Can also be trainvalids.txt')
#parser.add_argument('--num_caption',type=int, default=5, help='number of captions per image to be used for training')
parser.add_argument('--num_caption',type=int, default=10, help='number of captions per image to be used for training')
# parser.add_argument('--image_dir', type=str, default='images_th3', help='image directory in data')
parser.add_argument('--image_dir', type=str, default='images', help='image directory in data')
parser.add_argument('--flip',type=int, default=0, help='flip sentence')
parser.add_argument('--num_workers',type=int, default=4, help='num workers')
parser.add_argument('--mode',type=str, default='train', help='[train|test] mode')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
config = parser.parse_args()

loader = get_loader(config)
print('Batch size: {}'.format(config.batch_size))
print('The number of batches: {}'.format(len(loader)))

data_iter = iter(loader)
txt, img = next(data_iter)
print('Size of txt: [batch_size, doc_length, alphabet size]={}'.format(txt.size()))
print('Size of img: [batch_size, 1d image dim]={}'.format(img.size()))

'''
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--scale', '-s', type=int, default=1,
                    help='quantization scale factor')
best_prec1 = 0
'''

def main():
    global args
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
   
    train_dataset = Dataset(args)
    print('Dataset size: {}'.format(len(train_dataset)))
    data_loader = data.DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=(args.mode=='train'),
                                  num_workers=args.num_workers)
    
    model = CNNRNN_Char()
    if args.cuda:
        model = model.cuda()
#        model = torch.nn.DataParallel(model).cuda()
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
#    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    losses = AverageMeter()

#    model.conv2.register_forward_hook(printnorm)
#    model.conv3.register_forward_hook(printnorm)
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch)
        train(train_loader, model, optimizer, epoch)
#        test(test_loader, model, optimizer, losses, epoch)
#        load_eval(eval_loader,model,losses,epoch)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count +=n
        self.avg = self.sum /self.count

def get_loader(config):
    """Build and return a data loader."""
    dataset = Dataset(config)
    print('Dataset size: {}'.format(len(dataset)))

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.batch_size,
                                  shuffle=(config.mode=='train'),
                                  num_workers=config.num_workers)
    return data_loader

def printnorm(self, input, output):
    # input is a tuple of packed inputs
    # output is a Variable. output.data is the Tensor we are interested
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('')
    print('input: ', type(input))
    print('input[0]: ', type(input[0]))
    print('output: ', type(output))
    print('')
    print('input size:', input[0].size())
    print('output size:', output.data.size())
    print('output norm:', output.data.norm())


def save_checkpoint(state, is_best):
    filename='./ckpt_CNNRNN_char.pth.tar'
    torch.save(state, filename)
    if is_best:
        new_name = 'best_CNNRNN_char_' + str(best_prec1) + '.pth.tar'
        shutil.copyfile(filename, new_name)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 25))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(train_loader, model, optimizer,  epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
#        if batch_idx % args.log_interval == 0:
#            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                epoch, batch_idx * len(data), len(train_loader.dataset),
#                100. * batch_idx / len(train_loader), loss.data[0]))

def load_eval(test_loader, model, losses, epoch):
    global best_prec1, scale
    filename='./checkpoint_det_k32_s' + str(args.scale) + '.pth.tar'
    loaded_model = K32_BatchNet()
    loaded_model.cuda()
    ckp = torch.load(filename)
    loaded_model.load_state_dict(ckp['state_dict'])
    loaded_model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = loaded_model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        #correct += pred.eq(target.data.view_as(pred)).double().sum()
        correct += (pred==target.data.view_as(pred)).double().sum()
    test_loss /= len(test_loader.dataset)
    losses.update(test_loss, n=1)
    prec1 = 100. * correct / len(test_loader.dataset)

    print('Loaded Eval: Loss: {:.4f} ({:.4f}), Accuracy: {}/{} [{:.2f}%], epoch: {}, scale: {}'.format(
        losses.val, losses.avg, correct, len(test_loader.dataset),
        prec1, epoch, args.scale))
 
def test(test_loader, model, optimizer, losses, epoch):
    global best_prec1, scale
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
#        cprint(pred, 'red')
#        cprint(target, 'cyan')
#        cprint(str(target.data.view_as(pred)), 'red')
        correct += (pred==target.data.view_as(pred)).double().sum()
        #correct += pred.eq(target.data.view_as(pred)).double().sum()

    test_loss /= len(test_loader.dataset)
    losses.update(test_loss, n=1)
    prec1 = 100. * correct / len(test_loader.dataset)

    print('Test set: Loss: {:.4f} ({:.4f}), Accuracy: {}/{} [{:.2f}%], epoch: {}, scale: {}'.format(
        losses.val, losses.avg, correct, len(test_loader.dataset),
        prec1, epoch, args.scale))
    is_best = prec1 > best_prec1
    if is_best and best_prec1 != 0:
        os.remove('./det_k32_s' + str(args.scale) +'_best_' + str(best_prec1) + '.pth.tar')
    best_prec1 = max(prec1, best_prec1)
#    filename='./ckp_det_k32_s' + str(args.scale) + '_prec' + str(prec1) +'.pth.tar'
#    torch.save({'state_dict':model.state_dict()}, filename)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prect1':best_prec1,
    }, is_best)

if __name__ == '__main__':
    main()

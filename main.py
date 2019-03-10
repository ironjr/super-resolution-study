"""
main.py

Main routine for training and testing various network topology using CIFAR-10
dataset.

Jaerin Lee
Seoul National University
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as T

# Per-parameter settings for torch.optim.Optimizer
from util import group_weight

# Overcome laziness of managing checkpoints
from os import mkdir, path
from shutil import copy
from datetime import datetime

# External hyper-parameters and running environment settings
import argparse
import json
from copy import deepcopy


def main(args):
    # Number of epochs are determined with other three parameters if not directly
    # specified
    num_epochs = args.num_epochs
    if num_epochs is -1:
        num_epochs = (args.num_iters * args.batch_size + args.num_train - 1) \
                // args.num_train

    # Log for tensorboard statistics
    logger_train = None
    logger_val = None
    if args.use_tb and args.mode == 'train':
        from logger import Logger
        if args.label is None:
            dirname = './logs/' + datetime.today().strftime('%Y%m%d-%H%M%S')
        else:
            dirname = './logs/' + args.label
        if not path.isdir(dirname):
            mkdir(dirname)
        logger_train = Logger(dirname + '/train')
        logger_val = Logger(dirname + '/val')


    # Define transforms
    transform_train = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
    ])

    # Load CIFAR-10 dataset
    print('Loading datasets ...')
    dset_train = dset.ImageFolder('./datasets/train', transform=transform_train)
    loader_train = DataLoader(dset_train, batch_size=args.batch_size,
            shuffle=True, sampler=None) #sampler.SubsetRandomSampler(range(num_train)))
    loader_val = DataLoader(dset_train, batch_size=args.batch_size,
            shuffle=True, sampler=None) #sampler.SubsetRandomSampler(range(num_train, 50000)))
    dset_test = dset.ImageFolder('./datasets/test', transform=transform_test)
    loader_test = DataLoader(dset_test, batch_size=args.batch_size)
    print('Done!')

    # Set device (GPU or CPU)
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        # import torch.backends.cudnn as cudnn
        # cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    print('Using device:', device)


    # Set network model
    from models import vdsr
    model = vdsr.VDSR(
        imchannel=3,
        imsize=32,
        nfilter=64,
        nlayer=5
    )
    criterion = F.mse_loss

    # Define new optimizer specified by hyperparameters defined above
    optimizer = optim.SGD(group_weight(model),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay)

    # Load previous model
    if not args.try_new or args.mode == 'test':
        print('PyTorch is currently the loading model ... ', end='')

        model.load_state_dict(torch.load('model.pth'))
        model.cuda()

        print('Done!')

        # Optimizer is loaded when we continue training
        if args.mode == 'train':
            print('PyTorch is currently loading the optimizer ... ', end='')

            optimizer.load_state_dict(torch.load('optimizer.pth'))
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

            print('Done!')

    # Overwrite default hyperparameters for new run
    group_decay, group_no_decay = optimizer.param_groups
    group_decay['lr'] = args.lr
    group_decay['momentum'] = args.momentum
    group_decay['weight_decay'] = args.weight_decay
    group_no_decay['lr'] = args.lr
    group_no_decay['momentum'] = args.momentum

    # Train/Test the model
    from optimizer import train, test
    if args.mode == 'train':
        train(model, optimizer, args.scale_factor, criterion, loader_train,
                loader_val=loader_val, clip_grad=None, num_epochs=num_epochs,
                logger_train=logger_train, logger_val=logger_val,
                print_every=args.print_every, iteration_begins=args.iter_init)

        print('PyTorch is currently saving the model and the optimizer ...', end='')

        # Save model to checkpoint
        torch.save(model.state_dict(), 'model.pth')
        torch.save(optimizer.state_dict(), 'optimizer.pth')

        # Archive current model to checkpoints folder
        dirname = './checkpoints/' + datetime.today().strftime('%Y%m%d-%H%M%S')
        mkdir(dirname)
        copy('model.pth', dirname)
        copy('optimizer.pth', dirname)

        print('Done!')
    elif args.mode == 'test':
        test(model, loader_test)


# Define parser at the end of file for readibility
if __name__ == '__main__':
    # Parameterize the running envionment to run with a shell script
    parser = argparse.ArgumentParser(
            description='Run PyTorch on the classification problem.')
    parser.add_argument('--label', dest='label', type=str, default=None,
            help='name of run and folder where logs are to be stored')
    parser.add_argument('--scale-factor', dest='scale_factor', type=float,
            default=2.0, help='scale factor for super resolution')
    parser.add_argument('--mode', dest='mode', type=str, default='train',
            help='choose run mode between training and test')
    parser.add_argument('--schedule', dest='schedule_file', type=str, default=None,
            help='running schedule in json format')
    parser.add_argument('--use-tb', dest='use_tb', type = bool, default=True,
            help='use tensorboard logging')
    parser.add_argument('--try-new', dest='try_new', type=bool, default=True,
            help='choose whether use newly initialized model or not')
    parser.add_argument('--use-gpu', dest='use_gpu', type=bool, default=True,
            help='allow CUDA to accelerate run')
    parser.add_argument('--num-train', dest='num_train', type=int, default=50000,
            help='number of training set with maximum of 50000')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=1,
            help='batch size for training')
    parser.add_argument('--num-iters', dest='num_iters', type=int, default=1,
            help='number of iterations to train; will be overrided by \'--num-epochs\'')
    parser.add_argument('--num-epochs', dest='num_epochs', type=int, default=-1,
            help='number of epochs to train; overrides \'--num-iters\'')
    parser.add_argument('--iter-init', dest='iter_init', type=int, default=0,
            help='a point where iteration count begins for tensorboard stats')
    parser.add_argument('--print-every', dest='print_every', type=int, default=1,
            help='intermediate result evaluation period')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-1,
            help='learning rate for training')
    parser.add_argument('--weight-decay', dest='weight_decay', type=float,
            default=1e-4, help='weight decay for training with SGD optimizer')
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.9,
            help='momentum for training with SGD optimizer')
    args = parser.parse_args()

    # Get schedule from the json file
    # --- Structure of schedule.json --- #
    # list(dict(arguments) as event) as schedule
    # Arguments can be 'lr' 'weight_decay' 'momentum' 'num_iters' 'num_epochs'
    # 'mode'
    if args.schedule_file is not None:
        if '.json' in args.schedule_file:
            with open(args.schedule_file) as f:
                contents = json.load(f)

                iterations = 0
                iteration_begins = 0
                it_per_epoch = (args.num_train + args.batch_size - 1) // \
                        args.batch_size
                for event in contents['schedule']:
                    args_instance = deepcopy(args)
                    if 'mode' in event:
                        args_instance.mode = event['mode']
                    if 'label' in event:
                        args_instance.label = event['label']
                    if 'try_new' in event:
                        args_instance.try_new = event['try_new']
                        iteration_begins = 0
                    elif iteration_begins is not 0:
                        # Use existing model and optimizer after the first run
                        args_instance.try_new = False
                    if 'lr' in event:
                        args_instance.lr = event['lr']
                    if 'weight_decay' in event:
                        args_instance.weight_decay = event['weight_decay']
                    if 'momentum' in event:
                        args_instance.momentum = event['momentum']
                    if 'num_iters' in event:
                        args_instance.num_iters = event['num_iters']
                        iterations = ((event['num_iters'] + it_per_epoch - 1) // \
                                it_per_epoch) * it_per_epoch
                    if 'num_epochs' in event:
                        args_instance.num_epochs = event['num_epochs']
                        iterations = event['num_epochs'] * it_per_epoch

                    # Get the count of total iterations passed
                    args_instance.iter_init = iteration_begins

                    # Run once
                    main(args_instance)

                    # Post iteration count
                    iteration_begins += iterations

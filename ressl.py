import torch
from util.torch_dist_sum import *
from data.imagenet import *
from data.augmentation import *
from util.meter import *
from network.ressl import ReSSL
import time
import torch.nn as nn
import argparse
import math
import torch.nn.functional as F
import os

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=23457)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--t', type=float, default=0.04)
parser.add_argument('--backbone', type=str, default='resnet50')
args = parser.parse_args()
print(args)

epochs = args.epochs
warm_up = 5


def adjust_learning_rate(optimizer, epoch, base_lr, i, iteration_per_epoch):
    T = epoch * iteration_per_epoch + i
    warmup_iters = warm_up * iteration_per_epoch
    total_iters = (epochs - warm_up) * iteration_per_epoch

    if epoch < warm_up:
        lr = base_lr * 1.0 * T / warmup_iters
    else:
        T = T - warmup_iters
        lr = 0.5 * base_lr * (1 + math.cos(1.0 * T / total_iters * math.pi))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(train_loader, model, local_rank, rank, criterion, optimizer, base_lr, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    iteration_per_epoch = len(train_loader)

    end = time.time()
    for i, (img1, img2) in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, base_lr, i, iteration_per_epoch)
        # measure data loading time
        data_time.update(time.time() - end)

        if local_rank is not None:
            img1 = img1.cuda(local_rank, non_blocking=True)
            img2 = img2.cuda(local_rank, non_blocking=True)

        # compute output
        logitsq, ligitsk = model(im_q=img1, im_k=img2)
        loss = - torch.sum(F.softmax(ligitsk.detach() / args.t, dim=1) * F.log_softmax(logitsq / 0.1, dim=1), dim=1).mean()

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        losses.update(loss.item(), img1.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 20 == 0 and rank == 0:
            progress.display(i)


def main():
    from torch.nn.parallel import DistributedDataParallel
    from util.dist_init import dist_init
    
    rank, local_rank, world_size = dist_init(args.port)

    batch_size = 32 # single gpu
    num_workers = 8
    base_lr = args.lr

    model = ReSSL(backbone=args.backbone)
    model = DistributedDataParallel(model.to(local_rank), device_ids=[local_rank], output_device=local_rank)

    param_dict = {}
    for k, v in model.named_parameters():
        param_dict[k] = v

    bn_params = [v for n, v in param_dict.items() if ('bn' in n or 'bias' in n)]
    rest_params = [v for n, v in param_dict.items() if not ('bn' in n or 'bias' in n)]

    optimizer = torch.optim.SGD([{'params': bn_params, 'weight_decay': 0,},
                                    {'params': rest_params, 'weight_decay': 1e-4}],
                                    lr=base_lr, momentum=0.9, weight_decay=1e-4)

    torch.backends.cudnn.benchmark = True

    train_dataset = ImagenetContrastive(aug=[moco_aug, target_aug], max_class=1000)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    criterion = nn.CrossEntropyLoss().cuda(local_rank)
    
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    checkpoint_path = 'checkpoints/ressl-{}-{}.pth'.format(args.backbone, epochs)
    print('checkpoint_path:', checkpoint_path)
    if os.path.exists(checkpoint_path):
        checkpoint =  torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print(checkpoint_path, 'found, start from epoch', start_epoch)
    else:
        start_epoch = 0
        print(checkpoint_path, 'not found, start from epoch 0')
    

    model.train()
    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)
        train(train_loader, model, local_rank, rank, criterion, optimizer, base_lr, epoch)
        
        if rank == 0:
            torch.save(
                {
                    'model': model.state_dict(), 
                    'optimizer': optimizer.state_dict(), 
                    'epoch': epoch + 1
                }, checkpoint_path)
    

if __name__ == "__main__":
    main()

import time
import os
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torch.utils.data as data
from torchvision import transforms, datasets

import utils

parser = argparse.ArgumentParser()

# Data loading
parser.add_argument('--train_dir', required=True, type=str, 
        help='folder of training images')
parser.add_argument('--val_dir', required=True, type=str, 
        help='folder of validation images')
parser.add_argument('--gpu', type=int, default=1,
        help='use GPU if available')
parser.add_argument('--epochs', '-e', type=int, default=200, 
        help='max number of epochs')
parser.add_argument('--batch_size', '-N', type=int, default=32, 
        help='batch size')
parser.add_argument('--workers', type=int, default=1, 
        help='workers for data loading')

# Model
parser.add_argument('--compression_iters', '-i', type=int, default=8, 
        help='numer')

# Optimization
parser.add_argument('--lr', type=float, default=5e-4, 
        help='learning rate')

# Pretrained
parser.add_argument('--load_checkpoint', '-c', type=str, default=None,
        help='checkpoint from which to start training')
parser.add_argument('--load_optim', type=int, default=1,
        help='if loading checkpoint, also load optimizer')
parser.add_argument('--load_model_only', type=int, default=1,
        help='load only saved model, not checkpoint')

# Checkpoint/logging
parser.add_argument('--save_checkpoint', '-s', type=str, default='checkpoints', 
        help='directory to save checkpoints')
parser.add_argument('--log_every', type=str, default=None,
        help='how many iters between logging/printing')
parser.add_argument('--tensorboard', type=str, default=None,
        help='tensorboard log directory')


def main():
    global args
    args = parser.parse_args()

    # GPU 
    args.gpu = torch.cuda.is_available() and args.gpu

    # Log directory
    if args.tensorboard:
        try: 
            import tensorflow as tf
            args.tensorboard = tf.summary.FileWriter(args.tensorboard)
        except ImportError:
            print('Please install tensorflow to use tensorboard')
            args.tensorboard = None

    # Transformations 
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((32, 32)),
        transforms.RandomHorizontalFlip()
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # Datasets 
    train_set = datasets.ImageFolder(root=args.train_dir, 
            transform=train_transform)
    val_set = datasets.ImageFolder(root=args.val_dir,
            transform=val_transform

    # Dataloaders
   train_loader = data.DataLoader(
        dataset=train_set, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.workers,
    )
   val_loader = data.DataLoader(
        dataset=val_set, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers,
    ) 

    # Model
    encoder = network.EncoderCell()
    binarizer = network.Binarizer()
    decoder = network.DecoderCell()
    
    # Optimizer
    params = (list(encoder.parameters()) + 
              list(binarizer.parameters()) + 
              list(decoder.parameters()))
    optimizer = optim.Adam(params, lr=args.lr)
    scheduler = LS.MultiStepLR(solver, milestones=[3, 10, 20, 50, 100], gamma=0.5)

    # Init
    iteration = 0
    epoch = 0

    # Checkpoint
    if args.load_model_only:
        model.load_state_dict(torch.load(args.load_checkpoint))
    elif args.load_checkpoint:
        checkpoint = torch.load(args.load_checkpoint) # dictionary
        encoder.load_state_dict(checkpoint['encoder'])
        binarized.load_state_dict(checkpoint['binarizer'])
        decoder.load_state_dict(checkpoint['decoder'])
        if args.load_optim:
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                optimizer.load_state_dict(checkpoint['scheduler'])
            if 'iteration' in checkpoint and 'epoch' in checkpoint:
                iteration = checkpoint['iteration']
                epochs = checkpoint['epoch']
                scheduler.last_epoch = max(epoch - 1, 0)

    # GPU
    if args.gpu: 
        to_gpu = encoder, binarizer, decoder
        to_gpu = [x.cuda() for x in to_cuda]
        encoder, binarizer, decoder = to_gpu

    # Loop
    best_val = 0
    for epoch in range(epoch, args.epochs + 1):

        # Val
        #current_val = validate(val_loader, encoder, binarizer, decoder, args)
        #scheduler.step(current_val)

        # Tensorboard 
        #if args.tensorboard is not None:
        #    add_summary_value(args.tensorboard, 'validation', current_val, iteration)
        #    args.tensorboard.flush()

        # Checkpoint
        #if current_val > best_val:
        #    best_val = current_val
        #    utils.save(args.checkpoint, encoder, binarizer, decoder, 
        #               optimizer, scheduler, iteration, epoch)

        # Train
        iteration, epoch, loss = train(train_loader, encoder, binarizer, 
            decoder, optimizer, scheduler, epoch, iteration, args)

def train(train_loader, encoder, binarizer, decoder, optimizer,
          scheduler, epoch, iteration, args):
    '''Train model for a single epoch'''
    
    start_time = time.time()
    for i, (images, _) in enumerate(train_loader):
        images = images.cuda() if args.gpu else images
        data_time = time.time() - start_time

        # Create hidden states
        e_hidden_states = encoder.create_hidden()
        d_hidden_states = decoder.create_hidden()

        # Compress
        losses = []
        res = images
        for j in range(args.compression_iters):
            e_out, e_hidden_states = encoder(res, e_hidden_states)
            b_out = binarizer(e_out)
            d_out, d_hidden_states = decoder(b_out, d_hidden_states)
            res = res - d_out
            losses.append(res.abs().mean()) # mean absolute error
        
        # Backprop
        optimizer.zero_grad()
        loss = sum(losses) / args.iterations
        loss.backward()
        optimizer.step()
        compute_time = time.time() - data_time - start_time

        # Log
        iteration += 1
        if (iteration % args.log_every == 0):
            if args.tensorboard is not None:
                utils.write_summary(args.tensorboard, 'train_loss', loss, iteration)
                utils.write_summary(args.tensorboard, 'learning_rate', 
                        optimizer.current_lr, iteration)
                tf_summary_writer.flush()
            print('[Train] Epoch {e:4d} | \
                   Iter {i:6d} | \
                   Loss {l:6d.4f} | \
                   Compute time {ct:2f} | \
                   Data time {dt:2f} '.format(e=epoch, i=iteration, l=loss, 
                                              ct=compute_time, dt=data_time)

        return iteration, epoch + 1, loss

if __name__ == '__name__':
    main() 

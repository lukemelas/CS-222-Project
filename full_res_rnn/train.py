import time
import os
import argparse
import pdb

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
from torchvision import transforms, datasets

import utils
from models.network import EncoderCell, DecoderCell, Binarizer
from metric import MultiScaleSSIM

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
parser.add_argument('--batch_size', '-N', type=int, default=31, 
        help='batch size')
parser.add_argument('--workers', type=int, default=1, 
        help='workers for data loading')

# Model
parser.add_argument('--compression_iters', '-i', type=int, default=8, 
        help='number of residual compression iterations')

# Optimization
parser.add_argument('--lr', type=float, default=5e-4, 
        help='learning rate')

# Pretrained
parser.add_argument('--load_checkpoint', '-c', type=str, default=None,
        help='checkpoint from which to start training')
parser.add_argument('--load_optim', type=int, default=1,
        help='if loading checkpoint, also load optimizer')
parser.add_argument('--load_model_only', type=int, default=0,
        help='load only saved model, not checkpoint')

# Checkpoint/logging
parser.add_argument('--save_checkpoint', '-s', type=str, default='checkpoints', 
        help='directory to save checkpoints')
parser.add_argument('--log_every', type=int, default=20,
        help='how many iters between logging/printing')
parser.add_argument('--tensorboard', '-w', type=str, default=None,
        help='tensorboard log directory')

# Misc
parser.add_argument('--num_val_images', type=int, default=-1, 
        help='size of val set, -1 for all images')
parser.add_argument('--dir_save_images', type=str, default='output', 
        help='directory in which to save images') 
parser.add_argument('--num_save_images', type=int, default=0, 
        help='number of compressed images to save')


def main():
    global args
    args = parser.parse_args()
    print('ORIGINAL ARGS:  ', args.__dict__)
    start_time = time.time()

    # GPU 
    args.gpu = torch.cuda.is_available() and args.gpu
    if args.gpu:
        print('Using 1 GPU')

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
        #transforms.RandomResizedCrop(32),
        transforms.Resize((32,32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize((32,32)), #CenterCrop(32),
        transforms.ToTensor(),
        normalize,
    ])

    # Datasets 
    train_set = datasets.ImageFolder(root=args.train_dir, 
            transform=train_transform)
    val_set = datasets.ImageFolder(root=args.val_dir,
            transform=val_transform)

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
    encoder = EncoderCell()
    binarizer = Binarizer()
    decoder = DecoderCell()
    
    # Optimizer
    params = (list(encoder.parameters()) + 
              list(binarizer.parameters()) + 
              list(decoder.parameters()))
    optimizer = optim.Adam(params, lr=args.lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, 
        milestones=[3, 10, 20, 50, 100], gamma=0.5)

    # Init
    iteration = 0
    epoch = 0

    # Checkpoint
    if args.load_model_only:
        return NotImplementedError()
    elif args.load_checkpoint:
        checkpoint = torch.load(args.load_checkpoint)
        encoder.load_state_dict(checkpoint['encoder'])
        binarizer.load_state_dict(checkpoint['binarizer'])
        decoder.load_state_dict(checkpoint['decoder'])
        if args.load_optim:
            checkpoint = torch.load(args.load_checkpoint) # dictionary
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
        to_gpu = [x.cuda() for x in to_gpu]
        encoder, binarizer, decoder = to_gpu

    # Loop
    best_val = 0
    print('Beginning training. Setup time: {:.2f}s'.format(time.time() - start_time))
    for epoch in range(epoch, args.epochs + 1):

        # Val
        current_val = validate(val_loader, encoder, binarizer, decoder, args, epoch)
        scheduler.step(current_val)

        # Tensorboard 
        if args.tensorboard is not None:
            add_summary_value(args.tensorboard, 'validation', current_val, iteration)
            args.tensorboard.flush()

        # Checkpoint
        if current_val > best_val:
            best_val = current_val
            if args.save_checkpoint and args.save_checkpoint.lower() != 'none':
                fname = 'checkpoint_e{:03d}_v{:.3f}'.format(epoch, best_val)
                fname = os.path.join(args.save_checkpoint, fname)
                utils.save_checkpoint(fname, encoder, binarizer, 
                        decoder, optimizer, scheduler, iteration, epoch)
            print('[Valid] ~ new best ~ ')
        print()

        # Train
        iteration, epoch, loss = train(train_loader, encoder, binarizer, 
            decoder, optimizer, scheduler, epoch, iteration, args)

def train(train_loader, encoder, binarizer, decoder, optimizer,
          scheduler, epoch, iteration, args):
    '''Train model for a single epoch'''
    
    start_time = time.time()
    for i, (images, _) in enumerate(train_loader):
        batch_size = images.shape[0]
        images = images.cuda() if args.gpu else images
        data_time = time.time() - start_time

        # Create hidden states
        e_hidden_states = encoder.create_hidden(batch_size, gpu=args.gpu)
        d_hidden_states = decoder.create_hidden(batch_size, gpu=args.gpu)

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
        loss = sum(losses) / args.compression_iters 
        loss.backward()
        optimizer.step()
        compute_time = time.time() - data_time - start_time
        loss = loss.item()

        # Log
        iteration += 1
        if (iteration % args.log_every == 0):
            if args.tensorboard is not None:
                utils.write_summary(args.tensorboard, 'train_loss', loss, iteration)
                utils.write_summary(args.tensorboard, 'learning_rate', 
                        optimizer.current_lr, iteration)
                tf_summary_writer.flush()
            print(('[Train] Epoch {e:5d} '
                   '| Iter {i:6d} '
                   '| Loss {l:10.4f} '
                   '| Compute time {ct:8.2f} '
                   '| Data time {dt:8.2f} ').format(
                    e=epoch, i=iteration, l=loss, ct=compute_time, dt=data_time))

    return iteration, epoch + 1, loss

def get_metrics(original, compared):
    ssim = MultiScaleSSIM(original, compared, max_val=255)
    mse = np.mean(np.square(original - compared))
    psnr = np.clip(np.multiply(np.log10(
                255. * 255. / mse[mse > 0.]), 10.), 0., 99.99)[0]
    return ssim, psnr


def validate(val_loader, encoder, binarizer, decoder, args, epoch=None):
    '''Validate model'''
    start_time = time.time()
    N = args.num_val_images if args.num_val_images > 0 else len(val_loader) * args.batch_size
    
    # Metrics
    num_images = 0
    loss_total = 0
    ssim_total = 0
    psnr_total = 0

    # Block gradient
    with torch.no_grad():

        # Load images
        save_image_counter = 0
        for images, _ in val_loader:
            batch_size = images.shape[0]
            images = images.cuda() if args.gpu else images

            # Create hidden states
            e_hidden_states = encoder.create_hidden(batch_size, gpu=args.gpu)
            d_hidden_states = decoder.create_hidden(batch_size, gpu=args.gpu)

            # Compress
            codes = []
            res = images
            images_decoded = torch.zeros(images.shape)
            for j in range(args.compression_iters):
                e_out, e_hidden_states = encoder(res, e_hidden_states)
                b_out = binarizer(e_out)
                d_out, d_hidden_states = decoder(b_out, d_hidden_states)

                codes.append(b_out.data.cpu().numpy())
                res = res - d_out 
                images_decoded = images_decoded + d_out.data.cpu()

            # Calculate metrics: loss/SSIM/PSNR
            originals = images.cpu().data.numpy()
            compressed = images_decoded.cpu().data.numpy()
            loss = np.mean(np.abs(originals - compressed))
            ssim, psnr = get_metrics(originals, compressed)
            loss_total += loss
            ssim_total += ssim
            psnr_total += psnr
            num_images += originals.shape[0]

            # Save original/compressed images to file
            if save_image_counter < args.num_save_images: 
                for orig, comp in zip(images, images_decoded):
                    orig_fname = 'orig_{:05d}'.format(save_image_counter)
                    comp_fname = 'comp_{:05d}'.format(save_image_counter)
                    utils.save_image(orig, orig_fname, args, epoch=epoch)
                    utils.save_image(comp, comp_fname, args, epoch=epoch)
                    save_image_counter += 1
                    if save_image_counter >= args.num_save_images: 
                        break

            # End validation
            if args.num_val_images >= 0 and num_images >= args.num_val_images:
                break

            if num_images % args.log_every == 0:
                print(('[Valid] | Iter [{n:4d}/{N:4d}] '
                       '| Loss {l:10.4f} '
                       '| SSIM {s:8.3f} '
                       '| PSNR {p:8.3f} '
                       '| Time {t:6.2f}').format(
                          n=num_images, N=N,
                          l=loss_total/num_images, s=ssim_total/num_images,
                          p=psnr_total/num_images, t=time.time() - start_time))

        print(('[Valid] | Done [{n:4d}/{N:4d}] '
               '| Loss {l:10.4f} '
               '| SSIM {s:8.3f} '
               '| PSNR {p:8.3f} '
               '| Time {t:6.2f}').format(
                  n=num_images, N=N,
                  l=loss_total/num_images, s=ssim_total/num_images,
                  p=psnr_total/num_images, t=time.time() - start_time))

    return ssim_total / num_images

if __name__ == '__main__':
    main() 

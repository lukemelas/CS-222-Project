import os, time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

# Tensorboard for logging
try:
    import tensorflow as tf
except:
    tf = None


def save_checkpoint(path, encoder, binarizer, decoder, 
        optimizer=None, scheduler=None, iteration=None, epoch=None):
    '''Save model checkpoint to path'''
    checkpoint = {
        'encoder': encoder,
        'binarizer': binarizer,
        'decoder': decoder,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'iteration': iteration,
        'epoch': epoch
    }
    torch.save(checkpoint, path)
    return

def write_summary(writer, key, value, iteration):
    '''Write to tensorboard summary'''
    try: 
        summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
        writer.add_summary(summary, iteration)
    except:
        print('Error writing tf summary. Is tf installed?')
    return 


InvNormalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)

def save_image(img_tensor, fname, args, epoch=None):
    assert os.path.isdir(args.dir_save_images)
    fname = fname + ('_e{:03d}'.format(epoch) if epoch else '') + '.png'
    fname = os.path.join(args.dir_save_images, fname)
    img_tensor = InvNormalize(img_tensor)
    img = (img_tensor.cpu().data.numpy() * 255).astype(np.uint8).transpose(1,2,0)
    plt.imsave(arr=img, fname=fname)


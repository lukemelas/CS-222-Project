import os, time
import numpy as np
import torch

def save_checkpoint(path, encoder, binarizer, decoder, 
        optimizer=None, scheduler=None, iteration=None, epoch=None):
    checkpoint = {
        'encoder': encoder,
        'binarizer': binarizer,
        'decoder': decoder,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'iteration': iteration,
        'epoch': epoch
    }
    torch.save(path, checkpoint)
    return

def write_summary(writer, key, value, iteration):
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, iteration)
    return 




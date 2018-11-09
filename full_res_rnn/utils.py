import os, time
import numpy as np
import torch

try:
    import tensorflow as tf


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
    torch.save(path, checkpoint)
    return

def write_summary(writer, key, value, iteration):
    '''Write to tensorboard summary'''
    try: 
        summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
        writer.add_summary(summary, iteration)
    return 


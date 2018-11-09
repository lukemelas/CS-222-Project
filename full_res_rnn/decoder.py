import os
import argparse

import numpy as np
from scipy.misc import imread, imresize, imsave

import torch

import network

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, type=str, 
        help='path to model')
parser.add_argument('--input', required=True, type=str, 
        help='input codes')
parser.add_argument('--output', default='.', type=str, 
        help='output folder')
parser.add_argument('--gpu', type=int, default=1,
        help='use GPU if available')
parser.add_argument('--compression_iters', type=int, default=8,
        help='number of residual compression iterations')
args = parser.parse_args()

def main():

    # Load input codes
    input = np.load(args.input)
    codes = np.unpackbits(content['codes'])
    codes = np.reshape(codes, content['shape']).astype(np.float32) * 2 - 1
    codes = torch.from_numpy(codes)

    # Metadata
    iters, batch_size, C, H, W = codes.size()
    assert iters == args.compression_iters
    H, W = H*16, W*16

    # Model
    decoder = network.DecoderCell()
    checkpoint = torch.load(args.checkpoint)['decoder']
    decoder.load_state_dict(checkpoint)
    d_hidden_states = decoder.create_hidden(batch_size, gpu=args.gpu)

    # GPU
    if args.gpu:
        decoder = decoder.cuda()
        codes = codes.cuda()

    # Initial image
    image = torch.zeros(1, 3, H, W)
    
    # Block gradient
    decoder.eval()
    with torch.no_grad():

        # Decompress
        for i in args.compression_iters:
            d_out, d_hidden_states = decoder(codes[iters], d_hidden_states)
            image = image + output.data.cpu()
    
    # Save image
    fname = '{:02d}.png'.format(iters)
    fpath = os.path.join(args.output, fname)
    img_out = np.squeeze(image.numpy().clip(0,1))
    img_out = (img_out * 255.).astype(np.uint8).transpose(1,2,0)
    imsave(fpath, img_out)
    print('Decompressed image saved to {}'.format(fpath))

if __name__ == '__main__':
    main()

import argparse

import numpy as np
from scipy.misc import imread, imresize, imsave

import torch
from torch.autograd import Variable

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

    # Load input image
    image = imread(args.input, mode='RGB')
    image = torch.from_numpy(np.expand_dims(np.transpose(
                image.astype(np.float32) / 255.0, (2, 0, 1)), 0))

    # Metadata
    batch_size, C, H, W = image.size()
    assert H % 32 == 0 and W % 32 == 0
    
    # Model
    encoder = network.EncoderCell()
    binarizer = network.Binarizer()
    decoder = network.DecoderCell()
    checkpoint = torch.load(args.checkpoint)
    encoder.load_state_dict(checkpoint['encoder'])
    binarizer.load_state_dict(checkpoint['binarizer'])
    decoder.load_state_dict(checkpoint['decoder'])

    # GPU
    if args.gpu:
        tmp = encoder, binarizer, decoder, image
        tmp = [x.cuda() for x in tmp]
        encoder, binarizer, decoder, image = tmp

    # Hidden state
    e_hidden_states = encoder.create_hidden(batch_size, gpu=args.gpu)
    d_hidden_states = decoder.create_hidden(batch_size, gpu=args.gpu)
    
    # Block gradient
    encoder.eval()
    binarizer.eval()
    decoder.eval()
    with torch.no_grad():
        
        # Compress
        codes = []
        res = image
        for j in range(args.compression_iters):
            e_out, e_hidden_states = encoder(res, e_hidden_states)
            b_out = binarizer(e_out)
            d_out, d_hidden_states = decoder(b_out, d_hidden_states)
        res = res - d_out 
        codes.append(b_out.data.cpu().numpy())

        # Convert to bits and save
        codes = (np.stack(codes).astype(np.int8) + 1) // 2
        export = np.packbits(codes.reshape(-1))
        np.savez_compressed(args.output, shape=codes.shape, codes=export)

if __name__ == '__main__':
    main()

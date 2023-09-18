# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 23:08:33 2021

@author: susav
"""

import _pickle as pickle
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt

cifar10 = "../cifar-10-batches-py/"

parser = argparse.ArgumentParser("Plot training images in cifar10 dataset")
parser.add_argument("-i", "--image", type=int, default=0, 
                    help="Index of the image in cifar10. In range [0, 49999]")
args = parser.parse_args()


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def cifar10_plot(data, meta, im_idx=0):
    im = data[b'data'][im_idx, :]

    im_r = im[0:1024].reshape(32, 32)
    im_g = im[1024:2048].reshape(32, 32)
    im_b = im[2048:].reshape(32, 32)

    img = np.dstack((im_r, im_g, im_b))

    print("shape: ", img.shape)
    print("label: ", data[b'labels'][im_idx])
    print("category:", meta[b'label_names'][data[b'labels'][im_idx]])         

    plt.imshow(img) 
    plt.show()


def main():
    batch = (args.image // 10000) + 1
    idx = args.image - (batch-1)*10000

    data = unpickle(os.path.join(cifar10, "data_batch_" + str(batch)))
    meta = unpickle(os.path.join(cifar10, "batches.meta"))

    cifar10_plot(data, meta, im_idx=idx)


if __name__ == "__main__":
    main()
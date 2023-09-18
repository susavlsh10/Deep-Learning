# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 22:38:35 2021

@author: susav
"""
import numpy as np
import matplotlib.pyplot as plt
from DataLoader import load_data, train_valid_split, load_testing_images
from Configure import model_configs, training_configs
from ImageUtils import parse_record, parse_record_test, parse_record_train


x_train, y_train, x_test, y_test = load_data('../cifar-10-batches-py/')
x_train, y_train, x_valid, y_valid = train_valid_split(x_train, y_train)
key = np.load('predictions.npy')

x_test= x_test.astype('uint8')
# S= parse_record_train(x_test[1])

# plt.imshow(S)
# plt.show()

x_final = load_testing_images('../cifar-10-batches-py/')
idx = 20;


#A= parse_record(x_final[idx], 0)


Ax=x_final[idx]
#A = parse_record_train(Ax)
A = Ax.reshape((32,32,3))
plt.imshow(A)
plt.show()
print('predicted ans: {:d}'.format(np.argmax(key[idx])), end='\r', flush=True)


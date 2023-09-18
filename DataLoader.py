import os
import pickle
import numpy as np

"""This script implements the functions for reading data.
"""

def load_data(data_dir):
    """Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches
            are stored.

    Returns:
        x_train: An numpy array of shape [50000, 3072].
            (dtype=np.float32)
        y_train: An numpy array of shape [50000,].
            (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072].
            (dtype=np.float32)
        y_test: An numpy array of shape [10000,].
            (dtype=np.int32)
    """

    ### YOUR CODE HERE
    dirr=[] #all the directioary of the batch numbers
    dictio=[] #all the dictionary of the batches 
    
    batch_num=['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']
    
    for i in range(len(batch_num)):
      dirr.append(os.path.join(data_dir, batch_num[i]))
      with open(dirr[i], 'rb') as fo:
          dictio.append(pickle.load(fo, encoding='bytes'))
          
    x_train=np.concatenate((dictio[0][b'data'], dictio[1][b'data'], dictio[2][b'data'], dictio[3][b'data'], dictio[4][b'data']),axis=0)
    x_train=x_train.astype('float32')
    y_train=np.concatenate((dictio[0][b'labels'], dictio[1][b'labels'], dictio[2][b'labels'], dictio[3][b'labels'], dictio[4][b'labels']),axis=0)
    
    x_test=dictio[5][b'data']
    x_test=x_test.astype('float32')
    y_test=dictio[5][b'labels']
    ### END CODE HERE

    return x_train, y_train, x_test, y_test


def load_testing_images(data_dir):
    """Load the images in private testing dataset.

    Args:
        data_dir: A string. The directory where the testing images
        are stored.

    Returns:
        x_test: An numpy array of shape [N, 32, 32, 3].
            (dtype=np.float32)
    """

    ### YOUR CODE HERE
    _dir=os.path.join(data_dir, 'private_test_images_v3.npy')
    x_test=np.load(_dir)
    # x_test = x_test.reshape((x_test.shape[0],32,32,3))
    # x_test= x_test.astype('float32')
    
    
    ### END CODE HERE

    return x_test


def train_valid_split(x_train, y_train, train_ratio=0.8):
    """Split the original training data into a new training dataset
    and a validation dataset.

    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        train_ratio: A float number between 0 and 1.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    
    ### YOUR CODE HERE
    
    split_index= int(train_ratio * (x_train.shape[0]))
    
    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]
    ### END CODE HERE

    return x_train_new, y_train_new, x_valid, y_valid


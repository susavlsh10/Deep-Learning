### YOUR CODE HERE

import torch
import torch.nn as nn
import os, argparse
import numpy as np
from Model import MyModel
from DataLoader import load_data, train_valid_split, load_testing_images
from Configure import model_configs, training_configs
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("mode", help="train, test or predict")
parser.add_argument("data_dir", help="path to the data")
parser.add_argument("save_dir", help="path to save the results")
args = parser.parse_args()

use_cuda= torch.cuda.is_available()

if __name__ == '__main__':
    device = torch.device("cuda" if use_cuda else "cpu")
    model=MyModel(model_configs).to(device)

    if args.mode == 'train':
        x_train, y_train, x_test, y_test = load_data(args.data_dir)
        x_train, y_train, x_valid, y_valid = train_valid_split(x_train, y_train)

        model.train(x_train, y_train, training_configs, x_valid, y_valid)
        model.score(x_test, y_test)

    elif args.mode == 'test':
        _, _, x_test, y_test = load_data(args.data_dir)
        checkpointfile = os.path.join(model_configs['save_dir'], 'model-%d.ckpt'%(170))
        model.load(checkpointfile)        
        model.score(x_test, y_test)
        
    elif args.mode == 'predict': 
        checkpointfile = os.path.join(model_configs['save_dir'], 'model-%d.ckpt'%(170))
        model.load(checkpointfile)
        x_test = load_testing_images(args.data_dir)
        predictions_private, prob_private = model.predict_prob(x_test)
        
        softmax= nn.Softmax(dim=1)
        probabilities= softmax(prob_private)
        probabilities= probabilities.detach().numpy()

        np.save('predictions.npy', probabilities)
        if args.save_dir[len(args.save_dir)-3:len(args.save_dir)]!='npy':
            _dir= os.path.join(args.save_dir, 'predictions.npy')
        else:
            _dir=args.save_dir
        np.save(_dir, probabilities)
### END CODE HERE


### YOUR CODE HERE
import torch
import torch.nn as nn
import os, time
import numpy as np
from Network import MyNetwork
from ImageUtils import parse_record
from tqdm import tqdm


"""This script defines the training, validation and testing process.
"""

use_cuda= torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class MyModel(nn.Module):

    def __init__(self, configs):
        super(MyModel, self).__init__()
        self.configs = configs
        self.network = MyNetwork(configs)
        self.model_setup()


    def model_setup(self):
        self.criterion= nn.CrossEntropyLoss()
        self.optimizer= torch.optim.SGD(self.parameters(), lr=0.0001, momentum=0.997, weight_decay=self.configs['weight_decay'])

    def train(self, x_train, y_train, configs, x_valid=None, y_valid=None):
        max_epoch= configs['max_epoch']
        self.network.train()
        # Determine how many batches in an epoch
        num_samples = x_train.shape[0]
        num_batches = num_samples // self.configs['batch_size']
        self.scheduler= torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_batches, eta_min=0)
        batch_size=self.configs['batch_size']

        print('### Training... ###')
        for epoch in range(1, max_epoch+1):
            start_time = time.time()
            # Shuffle
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]

            # Set the learning rate for this epoch
            # Usage example: divide the initial learning rate by 10 after several epochs

            
            for i in range(num_batches+1):

                # Construct the current batch.
                if ((i*batch_size + batch_size)) > num_samples:
                  new_batch= num_samples - i*batch_size
                  batch_x=curr_x_train[i*batch_size:i*batch_size + new_batch,:]
                  batch_y=curr_y_train[i*batch_size:i*batch_size + new_batch]
                  batch_size=new_batch
                else:
                  batch_size=self.configs['batch_size']
                  batch_x=curr_x_train[i*batch_size:i*batch_size + batch_size,:]
                  batch_y=curr_y_train[i*batch_size:i*batch_size + batch_size]
                
                current_batch=np.ndarray(shape=(batch_size,3,32,32))
                
                for k in range(batch_size):
                    current_batch[k]=parse_record(batch_x[k], 1)
               
                current_batch=current_batch.astype('float32')
                batch_y=batch_y.astype('longlong')
                
                current_batch_x_tensor=torch.from_numpy(current_batch).to(device)
                current_batch_y_tensor=torch.from_numpy(batch_y).to(device)
                
                out=self.network(current_batch_x_tensor)
                loss= self.criterion(out, current_batch_y_tensor)
                # Don't forget to use "parse_record" to perform data preprocessing.
                # Don't forget L2 weight decay
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print('Batch {:d}/{:d} Loss {:.6f} \n' .format(i, num_batches+1, loss), end='\r', flush=True)
            self.scheduler.step()
            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.\n'.format(epoch, loss, duration))
                      
            if epoch % configs['save_interval'] == 0:
                check=[]
                check.append(epoch)
                self.save(epoch)
                self.evaluate(x_valid, y_valid, check)

    def save(self, epoch):
        checkpoint_path = os.path.join(self.configs['save_dir'], 'model-%d.ckpt'%(epoch))
        os.makedirs(self.configs['save_dir'], exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")
    
    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))

    def score(self, x, y):
      self.network.eval()
      num_samples=x.shape[0]
      num_batches = num_samples // self.configs['batch_size']
      batch_size=self.configs['batch_size']
      
      print('### Test or Validation ###')
      preds = []

      for i in tqdm(range(num_batches+1)):
        
        # Construct the current batch.
        if ((i*batch_size + batch_size)) > num_samples:
          new_batch= num_samples - i*batch_size
          batch_x=x[i*batch_size:i*batch_size + new_batch,:]
          batch_size=new_batch
        else:
          batch_x=x[i*batch_size:i*batch_size + batch_size,:]
          batch_size=self.configs['batch_size']

        current_batch=np.ndarray(shape=(batch_size,3,32,32))  
        for k in range(batch_size):
            current_batch[k]=parse_record(batch_x[k], 1)
        current_batch=current_batch.astype('float32')
        current_batch_x_tensor=torch.from_numpy(current_batch).to(device)
        out=self.network(current_batch_x_tensor)
        
        for j in range(batch_size):
            preds.append(out[j].argmax())
            
      y = torch.tensor(y[0:len(preds)])
      preds = torch.tensor(preds)
      print('Test accuracy: {:.4f}'.format(torch.sum(preds==y)/y.shape[0]))
    
    def evaluate(self, x, y, checkpoint_num_list):
        self.network.eval()
        num_samples=x.shape[0]
        num_batches = num_samples // self.configs['batch_size']
        batch_size=self.configs['batch_size']
        
        print('### Test or Validation ###')
        for checkpoint_num in checkpoint_num_list:
            if checkpoint_num_list:
              checkpointfile = os.path.join(self.configs['save_dir'], 'model-%d.ckpt'%(checkpoint_num))
              self.load(checkpointfile)
            preds = []
            batch_size=self.configs['batch_size']
            for i in tqdm(range(num_batches+1)):

              ### YOUR CODE HERE
              # Construct the current batch.
              if ((i*batch_size + batch_size)) > num_samples:
                new_batch= num_samples - i*batch_size
                batch_x=x[i*batch_size:i*batch_size + new_batch,:]
                batch_size=new_batch
              else:
                batch_x=x[i*batch_size:i*batch_size + batch_size,:]
                batch_size=self.configs['batch_size']

              current_batch=np.ndarray(shape=(batch_size,3,32,32))  
              for k in range(batch_size):
                  current_batch[k]=parse_record(batch_x[k], 1)
              current_batch=current_batch.astype('float32')
              current_batch_x_tensor=torch.from_numpy(current_batch).to(device)
              out=self.network(current_batch_x_tensor)
              
              for j in range(batch_size):
                  preds.append(out[j].detach().cpu().argmax())
                
                ### END CODE HERE

            y = torch.tensor(y[0:len(preds)])
            preds = torch.tensor(preds)
            print('Test accuracy: {:.4f}'.format(torch.sum(preds==y)/y.shape[0]))

    def predict_prob(self, x):
        self.network.eval()
        num_samples=x.shape[0]
        num_batches = num_samples // self.configs['batch_size']
        batch_size=self.configs['batch_size']
        
        print('### Test or Validation ###')
        preds = []
        prob=torch.zeros([num_samples, 10])
        for i in tqdm(range(num_batches+1)):
            ### YOUR CODE HERE
            # Construct the current batch.
            
            if ((i*batch_size + batch_size)) > num_samples:
              new_batch= num_samples - i*batch_size
              batch_x=x[i*batch_size:i*batch_size + new_batch,:]
              batch_size=new_batch
            else:
              batch_x=x[i*batch_size:i*batch_size + batch_size,:]
            current_batch=np.ndarray(shape=(batch_size,3,32,32))
            
            for k in range(batch_size):
                current_batch[k]=parse_record(batch_x[k], 1)
            current_batch=current_batch.astype('float32')
            current_batch_x_tensor=torch.from_numpy(current_batch).to(device)
            out=self.network(current_batch_x_tensor)
            
            if ((i*self.configs['batch_size'] + self.configs['batch_size'])) > num_samples:
              prob[i*self.configs['batch_size']:i*self.configs['batch_size'] + new_batch]=out.detach().cpu()
            else:
              prob[i*batch_size:i*batch_size + batch_size]=out.detach().cpu()
            
            for j in range(batch_size):
              #prob[j]=out[j].detach().cpu()
              preds.append(out[j].argmax())
            
        predictions=np.asarray(preds)
        return predictions, prob


### END CODE HERE
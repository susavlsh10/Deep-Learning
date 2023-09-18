#@title
### YOUR CODE HERE
import torch
import torch.nn as nn 
from torch.functional import Tensor

"""This script defines the network.
"""
ELU= nn.ELU()

class MyNetwork(nn.Module):
    def __init__(self, configs):

        super(MyNetwork, self).__init__()
        
        self.resnet_version = configs['version']
        self.resnet_size = configs['resnet_size']
        self.num_classes = configs['num_classes']
        self.first_num_filters = configs['first_num_filters']
        
        ### YOUR CODE HERE
        # define conv1  filters, strides, first_num_filters
        self.start_layer=start_block(filters=self.first_num_filters, first_num_filters=3)  
        #self.start_layer=nn.Conv2d(3, self.first_num_filters, kernel_size=7, stride=1, padding=3)
        
        
        ### YOUR CODE HERE
        if self.resnet_version == 1:
            self.batch_norm_relu_start = batch_norm_relu_layer(
                num_features=self.first_num_filters, 
                eps=1e-5, 
                momentum=0.997,
            )
        if self.resnet_version == 1:
            block_fn = standard_block_SE
        else:
            block_fn = bottleneck_block_SE

        self.stack_layers = nn.ModuleList()
        for i in range(3):
            filters = self.first_num_filters * (2**i)
            strides = 1 if i == 0 else 2
            self.first_num_filters = 2*self.first_num_filters if i==2 else self.first_num_filters
            self.stack_layers.append(stack_layer(filters, block_fn, strides, self.resnet_size[i], self.first_num_filters))
        
        self.output_layer = output_layer(filters*4, self.resnet_version, self.num_classes)
    
    def forward(self, inputs):
        outputs = self.start_layer(inputs)
        if self.resnet_version == 1:
            outputs = self.batch_norm_relu_start(outputs)
        for i in range(3):
            outputs = self.stack_layers[i](outputs)
        outputs = self.output_layer(outputs)
        return outputs

class batch_norm_relu_layer(nn.Module):
    """ Perform batch normalization then relu.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.997) -> None:
        super(batch_norm_relu_layer, self).__init__()
        ### YOUR CODE HERE
        self.BN=nn.BatchNorm2d(num_features, eps, momentum)
        self.relu=nn.ELU()
        ### YOUR CODE HERE
        
    def forward(self, inputs: Tensor) -> Tensor:
        ### YOUR CODE HERE
        out=self.BN(inputs)
        out=self.relu(out)
        
        return out

class squeeze_expand(nn.Module):
    def __init__(self, channels, r=16):
        super(squeeze_expand, self).__init__()
        self.Squeeze = nn.AdaptiveAvgPool2d(1)
        self.Expand = nn.Sequential(
            nn.Linear(channels, channels // r, bias=False),
            nn.ELU(),
            nn.Linear(channels // r, channels, bias= False),
            nn.Sigmoid()
            )
        
    def forward(self, X):
        
        batch_size, channels, _, _ = X.shape
        out = self.Squeeze(X).view(batch_size, channels)
        out = self.Expand(out).view(batch_size, channels, 1, 1)
        
        return X * out.expand_as(X)
    
    
class start_block(nn.Module):
     def __init__(self, filters, first_num_filters) -> None:
         super(start_block, self).__init__()
         self.conv1= nn.Conv2d(first_num_filters, filters, kernel_size=3, stride=1, padding=1) 
         self.conv2= nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
         self.conv3= nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
         self.batch_norm_relu_start = batch_norm_relu_layer(num_features=filters, eps=1e-5, momentum=0.997)
     def forward(self, inputs: Tensor) ->Tensor:
         out = self.conv1(inputs)
         out= self.batch_norm_relu_start(out)
         
         out = self.conv2(out)
         out= self.batch_norm_relu_start(out)

         out = self.conv3(out)
         out= self.batch_norm_relu_start(out)         
         return out
     
class standard_block_SE(nn.Module):
    def __init__(self, filters, projection_shortcut, strides, first_num_filters) -> None:
        super(standard_block_SE, self).__init__()
        
        if projection_shortcut is not None: #first block
            self.conv1=nn.Conv2d(first_num_filters, filters, kernel_size=3, stride=strides, padding=1)
        else:
            self.conv1=nn.Conv2d(first_num_filters, filters, kernel_size=3, stride=1, padding=1)
            
        self.conv2=nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        
        self.batch_norm_relu_start = batch_norm_relu_layer(num_features=filters, eps=1e-5, momentum=0.997)
        self.BN=nn.BatchNorm2d(filters, 1e-5, 0.997)
        self.relu=nn.ELU()
        self.projection_shortcut = projection_shortcut       
        
        self.SE=squeeze_expand(filters)
        
        if self.projection_shortcut is not None:
            self.shortcut=nn.Conv2d(first_num_filters, filters, kernel_size=1, stride=strides)          
        
    def forward(self, inputs: Tensor) -> Tensor:
        
        if self.projection_shortcut is not None:
            shortcut=self.shortcut(inputs)
        else:
            shortcut=inputs
            
        out=self.conv1(inputs)
        out=self.batch_norm_relu_start(out)
        
        out=self.conv2(out)
        out=self.BN(out)

        out= self.SE(out)   
        
        out += shortcut
        out=self.relu(out)
        return out
    
class bottleneck_block_SE(nn.Module):

    def __init__(self, filters, projection_shortcut, strides, first_num_filters) -> None:
        super(bottleneck_block_SE, self).__init__()
        
        ### YOUR CODE HERE

        self.projection_shortcut=projection_shortcut
        if projection_shortcut is not None: #first block
            if filters>=128:
                in_chan=int(filters/2)
            else:
                in_chan=first_num_filters
            self.BN_relu1=batch_norm_relu_layer(num_features=in_chan, eps=1e-5, momentum=0.997)
            self.conv1= nn.Conv2d(in_channels=in_chan, out_channels=int(filters/4), kernel_size=1, stride=strides)
            self.shortcut=nn.Conv2d(in_chan, filters, kernel_size=1, stride=strides)
        else:
            self.BN_relu1=batch_norm_relu_layer(num_features=filters, eps=1e-5, momentum=0.997)
            self.conv1= nn.Conv2d(in_channels=filters, out_channels=int(filters/4), kernel_size=1)
            
        self.BN_relu2=batch_norm_relu_layer(num_features=int(filters/4), eps=1e-5, momentum=0.997)
        self.conv2= nn.Conv2d(in_channels=int(filters/4), out_channels=int(filters/4), kernel_size=3, stride=1, padding=1)
        
        self.BN_relu3=batch_norm_relu_layer(num_features=int(filters/4), eps=1e-5, momentum=0.997)
        self.conv3= nn.Conv2d(in_channels=int(filters/4), out_channels=filters, kernel_size=1)      
        
        self.SE= squeeze_expand(filters)
    
    def forward(self, inputs: Tensor) -> Tensor:
        ### YOUR CODE HERE
        # The projection shortcut should come after the first batch norm and ReLU
    		#since it performs a 1x1 convolution.
        out=self.BN_relu1(inputs)
        if self.projection_shortcut is not None:
            shortcut=self.shortcut(out)
        else:
            shortcut=out #the projection shortcut after the first BN and RELU
        
        out=self.conv1(out) 
        out=self.BN_relu2(out)
        
        out=self.conv2(out)
        out=self.BN_relu3(out)
        
        out=self.conv3(out)

        out= self.SE(out)                
    
        out += shortcut
        
        return out

        ### YOUR CODE HERE
    
class stack_layer(nn.Module):

    def __init__(self, filters, block_fn, strides, resnet_size, first_num_filters) -> None:
        super(stack_layer, self).__init__()
        filters_out = filters * 4 if block_fn is bottleneck_block_SE else filters
        ### END CODE HERE
        # projection_shortcut = ?
        # Only the first block per stack_layer uses projection_shortcut and strides
        self.layers=nn.ModuleList() #store all the blocks in this list
        self.resnet_size=resnet_size
        for i in range(resnet_size):
            if i ==0:
                projection_shortcut=True
                filters_first=first_num_filters
            else:
                projection_shortcut=None
                filters_first=filters
            
            if block_fn == standard_block_SE:
                block=standard_block_SE(filters_out, projection_shortcut, strides, filters_first)
                #add else
            else:
                block=bottleneck_block_SE(filters_out, projection_shortcut, strides, filters_first)
            self.layers.append(block)
        ### END CODE HERE
    
    def forward(self, inputs: Tensor) -> Tensor:
        ### END CODE HERE
        out=inputs
        for j in range(self.resnet_size):
            out=self.layers[j](out)
        return out
        ### END CODE HERE

class output_layer(nn.Module):
    """ Implement the output layer.

    Args:
        filters: A positive integer. The number of filters.
        resnet_version: 1 or 2, If 2, use the bottleneck blocks.
        num_classes: A positive integer. Define the number of classes.
    """
    def __init__(self, filters, resnet_version, num_classes) -> None:
        super(output_layer, self).__init__()
        # Only apply the BN and ReLU for model that does pre_activation in each
		# bottleneck block, e.g. resnet V2.
        if (resnet_version == 2):
            self.bn_relu = batch_norm_relu_layer(filters, eps=1e-5, momentum=0.997)
        
        ### END CODE HERE
        if (resnet_version==1):
            filters=int(filters/4)
        self.resnet_version=resnet_version
        self.filters=filters
        
        #self.Avg= nn.AvgPool2d(2, stride=2)
        self.fc= nn.Linear(filters*8*8, filters*4*4)
        self.drop=nn.Dropout(0.25)
        self.fc1= nn.Linear(filters*4*4, filters*2*2)
        self.relu=nn.ELU()
        self.fc2= nn.Linear(filters*2*2, filters)
        self.fc3= nn.Linear(filters, num_classes)
        self.Softmax= nn.Softmax(dim=1)
        ### END CODE HERE
    
    def forward(self, inputs: Tensor) -> Tensor:
        ### END CODE HERE
        out=inputs
        if (self.resnet_version == 2):
            out=self.bn_relu(out)
        out=out.view(-1, self.filters * 8*8)
        out=self.drop(out)
        out=self.fc(out)
        out=self.relu(out)

        out=self.drop(out)
        out=self.fc1(out)
        out=self.relu(out)
        
        out=self.drop(out)
        out=self.fc2(out)
        out=self.relu(out)
        
        out=self.drop(out)
        out=self.fc3(out)

        return out
        ### END CODE HERE
### END CODE HERE

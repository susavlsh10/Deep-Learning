# Below configures are examples, 
# you can modify them as you wish.

### YOUR CODE HERE

model_configs = {
	"name": 'MyModel',
	"save_dir": '../saved_models/',
	"depth": 2, 
	"weight_decay": 2e-4,
    "version": 2,
    "resnet_size": [18, 18, 18],
    "num_classes":10,
    "first_num_filters": 16,
    "batch_size": 128,
    "save_interval": 10,
    "max_epoch": 200,
    "data_dir": '../cifar-10-batches-py/'
}


training_configs = {
	"learning_rate": 0.0001,
    "max_epoch": 200,
    "batch_size": 128,
    "save_interval": 10
	# ...
}

### END CODE HERE
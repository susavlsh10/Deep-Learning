# Deep-Learning

How to run the code

To execute the program, the user needs to run main.py with the following command line parameters: main.py <mode> <data_dir> <save_dir>
• <mode> is either ‘train’, ‘test’, or ‘predict’.
• <data_dir> is the directory where the training and test data are stored. An exampleof a valid directory: ‘../cifar-10-batches-py/’. 
• <save_dir> is where the directory where the user wants to store the predictions onthe private test images. An example of a valid directory: ‘../saved_models/’

For example, use ‘main.py train ../cifar-10-batches-py/ ../saved_models/’ to train the net-work.

To test the model, use ‘main.py test ../cifar-10-batches-py/ ../saved_models/’

To predict on the private dataset, use ‘main.py predict../cifar-10-batches-py/ ../saved_models/’

In the examples shown above, the assumption is that all the training and test dataset including the private dataset are stored in the folder  'cifar-10-batches-py'.

The submission consists of a checkpoint file named “model-170.ckpt” in the “saved_models”directory.
This checkpoint file consists of a pretrained network which is used in the modes:‘test’ and ‘predict’. 
By default, in modes ‘test’ and ‘predict’, the code will load the checkpointfile from the directory ‘../saved_models/’, and do the computations.

The CIFAR 10 dataset needs to be stored in a directory named: 'cifar-10-batches-py' in an uncompressed format on the same directory as the folder 'code'.
It is also recommended to store the private test images on the same directoy as the CIFAR 10 dataset.

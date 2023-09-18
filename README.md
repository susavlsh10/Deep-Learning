# ResNetX implementation

This repository summarizes a deep learning model to perform 10-class image classification using the CIFAR-10 dataset. The model uses a variant of the ResNet architecture with Squeeze and Expand layers, ELU activation function, dropouts, extended input and linear layers. The highest accuracy achieved using the bottleneck architecture on the test data was 88.9%

The code files are structured as follows:

- **main.py**: Includes the code that loads the dataset and performs the training, testing and prediction.
- **DataLoader.py**: Includes the code that defines functions related to data I/O.
- **ImageUtils.py**: Includes the code that defines functions for any (pre-)processing of the images.
- **Configure.py**: Includes dictionaries that set the model configurations, hyper-parameters, training settings, etc. The dictionaries are imported to main.py
- **Model.py**: Includes the code that defines the your model in a class. The class is initialized with the configuration dictionaries and should have at least the methods “train(X, Y, configs,[X_valid, Y_valid,])”, “evaluate(X, Y)”, “predict_prob(X)”. The defined model class is imported to and referenced in main.py.
- **Network.py**: Includes the code that defines the network architecture. The defined
network will be imported and referenced in Model.py.

# How to run the code

To execute the program, the user needs to run main.py with the following command line parameters: main.py </mode/> <data_dir> <save_dir>
- </mode/> is either ‘train’, ‘test’, or ‘predict’.
- <data_dir> is the directory where the training and test data are stored. An exampleof a valid directory: ‘../cifar-10-batches-py/’. 
- <save_dir> is where the directory where the user wants to store the predictions onthe private test images. An example of a valid directory: ‘../saved_models/’

For example, use ‘main.py train ../cifar-10-batches-py/ ../saved_models/’ to train the net-work.

To test the model, use ‘main.py test ../cifar-10-batches-py/ ../saved_models/’

To predict on the private dataset, use ‘main.py predict../cifar-10-batches-py/ ../saved_models/’

The CIFAR 10 dataset needs to be stored in an uncompressed format.

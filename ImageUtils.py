import numpy as np

"""This script implements the functions for data augmentation
and preprocessing.
"""

def parse_record(record, training):
    """Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [32, 32, 3].
    """
    ### YOUR CODE HERE
    # Reshape from [depth * height * width] to [depth, height, width].
    depth_major = record.reshape((3, 32, 32))
    
    # Convert from [depth, height, width] to [height, width, depth]
    image = np.transpose(depth_major, [1, 2, 0])
    ### END CODE HERE

    image = preprocess_image(image, training) # If any.

    # Convert from [height, width, depth] to [depth, height, width]
    image = np.transpose(image, [2, 0, 1])
    
    return image

def parse_record_train(record):
    """Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [32, 32, 3].
    """
    ### YOUR CODE HERE
    # Reshape from [depth * height * width] to [depth, height, width].
    
    im_r = record[0:1024].reshape(32, 32)
    im_g = record[1024:2048].reshape(32, 32)
    im_b = record[2048:].reshape(32, 32)

    img = np.dstack((im_r, im_g, im_b))
    
    return img

def parse_record_test(record):
    """Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [32, 32, 3].
    """
    ### YOUR CODE HERE
    # Reshape from [depth * height * width] to [depth, height, width].
    A = record.reshape((32, 32, 3))

    image = np.zeros([3,32,32])

    image[0]= A[:,:,0]
    image[1]= A[:,:,1]
    image[2]= A[:,:,2]

    return image

def preprocess_image(image, training):
    """Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [32, 32, 3].
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [32, 32, 3]. The processed image.
    """

    if training:

        # Resize the image to add four extra pixels on each side.
        
        image_x0=np.pad(image[:,:,0],(4,4)) #zero padding on each side
        image_x1=np.pad(image[:,:,1],(4,4))
        image_x2=np.pad(image[:,:,2],(4,4))

        # Randomly crop a [32, 32] section of the image.
        rand_row=np.random.randint(0,8)
        rand_col=np.random.randint(0,8)
        image_x0_crop=image_x0[rand_row:rand_row+32,rand_col:rand_col+32]
        image_x1_crop=image_x1[rand_row:rand_row+32,rand_col:rand_col+32]
        image_x2_crop=image_x2[rand_row:rand_row+32,rand_col:rand_col+32]

        # Randomly flip the image horizontally.
        rand_flip=np.random.randint(0,3)
        if rand_flip == 1:
            image_x0_crop=np.flip(image_x0_crop, axis=1)
            image_x1_crop=np.flip(image_x1_crop, axis=1)
            image_x2_crop=np.flip(image_x2_crop, axis=1)

        # Subtract off the mean and divide by the standard deviation of the pixels.
        image_x0= (image_x0_crop-np.mean(image_x0_crop))/np.std(image_x0_crop)
        image_x1= (image_x1_crop-np.mean(image_x1_crop))/np.std(image_x1_crop)
        image_x2= (image_x2_crop-np.mean(image_x2_crop))/np.std(image_x2_crop)
        
        image=np.stack([image_x0, image_x1, image_x2], axis=2) #combining the three dimensions
    
    else:
        image=image

    return image


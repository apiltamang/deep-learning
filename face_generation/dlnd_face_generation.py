
# coding: utf-8

# # Face Generation
# In this project, you'll use generative adversarial networks to generate new images of faces.
# ### Get the Data
# You'll be using two datasets in this project:
# - MNIST
# - CelebA
# 
# Since the celebA dataset is complex and you're doing GANs in a project for the first time, we want you to test your neural network on MNIST before CelebA.  Running the GANs on MNIST will allow you to see how well your model trains sooner.
# 
# If you're using [FloydHub](https://www.floydhub.com/), set `data_dir` to "/input" and use the [FloydHub data ID](http://docs.floydhub.com/home/using_datasets/) "R5KrjnANiKVhLWAkpXhNBe".

# In[1]:

data_dir = './data'

# FloydHub - Use with data ID "R5KrjnANiKVhLWAkpXhNBe"
#data_dir = '/input'

from tensorflow.python import debug as tf_debug


def has_inf_or_nan(datum, tensor):
  return np.any(np.isnan(tensor)) or np.any(np.isinf(tensor))

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper

helper.download_extract('mnist', data_dir)
helper.download_extract('celeba', data_dir)


# ## Explore the Data
# ### MNIST
# As you're aware, the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset contains images of handwritten digits. You can view the first number of examples by changing `show_n_images`. 

# In[2]:

show_n_images = 25

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# get_ipython().magic('matplotlib inline')
import os
from glob import glob
# from matplotlib import pyplot

mnist_images = helper.get_batch(glob(os.path.join(data_dir, 'mnist/*.jpg'))[:show_n_images], 28, 28, 'L')
#pyplot.imshow(helper.images_square_grid(mnist_images, 'L'), cmap='gray')


# ### CelebA
# The [CelebFaces Attributes Dataset (CelebA)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset contains over 200,000 celebrity images with annotations.  Since you're going to be generating faces, you won't need the annotations.  You can view the first number of examples by changing `show_n_images`.

# In[3]:

#show_n_images = 25
#
#"""
#DON'T MODIFY ANYTHING IN THIS CELL
#"""
#mnist_images = helper.get_batch(glob(os.path.join(data_dir, 'img_align_celeba/*.jpg'))[:show_n_images], 28, 28, 'RGB')
#pyplot.imshow(helper.images_square_grid(mnist_images, 'RGB'))


# ## Preprocess the Data
# Since the project's main focus is on building the GANs, we'll preprocess the data for you.  The values of the MNIST and CelebA dataset will be in the range of -0.5 to 0.5 of 28x28 dimensional images.  The CelebA images will be cropped to remove parts of the image that don't include a face, then resized down to 28x28.
# 
# The MNIST images are black and white images with a single [color channel](https://en.wikipedia.org/wiki/Channel_(digital_image%29) while the CelebA images have [3 color channels (RGB color channel)](https://en.wikipedia.org/wiki/Channel_(digital_image%29#RGB_Images).
# ## Build the Neural Network
# You'll build the components necessary to build a GANs by implementing the following functions below:
# - `model_inputs`
# - `discriminator`
# - `generator`
# - `model_loss`
# - `model_opt`
# - `train`
# 
# ### Check the Version of TensorFlow and Access to GPU
# This will check to make sure you have the correct version of TensorFlow and access to a GPU

# In[4]:

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


# ### Input
# Implement the `model_inputs` function to create TF Placeholders for the Neural Network. It should create the following placeholders:
# - Real input images placeholder with rank 4 using `image_width`, `image_height`, and `image_channels`.
# - Z input placeholder with rank 2 using `z_dim`.
# - Learning rate placeholder with rank 0.
# 
# Return the placeholders in the following the tuple (tensor of real input images, tensor of z data)

# In[5]:

import problem_unittests as tests

def model_inputs(image_width, image_height, image_channels, z_dim):
    """
    Create the model inputs
    :param image_width: The input image width
    :param image_height: The input image height
    :param image_channels: The number of image channels
    :param z_dim: The dimension of Z
    :return: Tuple of (tensor of real input images, tensor of z data, learning rate)
    """
    # TODO: Implement Function

    input_real = tf.placeholder(dtype=tf.float32, shape=(None, image_width, image_height, image_channels))
    input_z = tf.placeholder(dtype=tf.float32, shape=(None, z_dim))
    learn_rate = tf.placeholder(dtype=tf.float32, shape=(None))
    return input_real, input_z, learn_rate


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
#tests.test_model_inputs(model_inputs)


# ### Discriminator
# Implement `discriminator` to create a discriminator neural network that discriminates on `images`.  This function should be able to reuse the variabes in the neural network.  Use [`tf.variable_scope`](https://www.tensorflow.org/api_docs/python/tf/variable_scope) with a scope name of "discriminator" to allow the variables to be reused.  The function should return a tuple of (tensor output of the discriminator, tensor logits of the discriminator).

# In[20]:

def discriminator(images, reuse=False, alpha=0.2, is_training=True):
    """
    Create the discriminator network
    :param image: Tensor of input image(s)
    :param reuse: Boolean if the weights should be reused
    :return: Tuple of (tensor output of the discriminator, tensor logits of the discriminator)
    """

    with tf.variable_scope("discriminator", reuse=reuse):
        
        # input: 28 * 28 * num_channels    
        x1 = tf.layers.conv2d(inputs=images, filters=128, kernel_size=5, strides=2, activation=None, padding='same', name="x1.conv2d")
        x1 = tf.maximum(alpha * x1, x1, name="x1.leaky_relu")

        # input: 14 * 14 * 128   
        x2 = tf.layers.conv2d(inputs=x1, filters=256, kernel_size=5, strides=2, activation=None, padding='same', name="x2.conv2d")
        x2 = tf.layers.batch_normalization(inputs=x2, training=is_training, name="x2.bn")
        x2 = tf.maximum(alpha * x2, x2, name="x2.leaky_relu")

        # input: 14 * 14 * 256    
        x3 = tf.layers.conv2d(inputs=x2, filters=512, kernel_size=5, strides=2, activation=None, padding='valid', name="x3.conv2d")
        x3 = tf.layers.batch_normalization(inputs=x3, training=is_training, name="x3.bn")
        x3 = tf.maximum(alpha * x3, x3, name="x3.leaky_relu")

        # input: 2 * 2 * 512
        x4 = tf.reshape(tensor=x3, shape=(-1, 2*2*512))
        
        # input: 2048
        x4 = tf.layers.dense(inputs=x4, units=1, activation=None, name="x4.dense")
        logits = tf.sigmoid(x4, name="x4.sigmoid")

        return x4, logits

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
#tests.test_discriminator(discriminator, tf)


# ### Generator
# Implement `generator` to generate an image using `z`. This function should be able to reuse the variabes in the neural network.  Use [`tf.variable_scope`](https://www.tensorflow.org/api_docs/python/tf/variable_scope) with a scope name of "generator" to allow the variables to be reused. The function should return the generated 28 x 28 x `out_channel_dim` images.

# In[21]:

def generator(z, out_channel_dim, reuse=False, alpha=0.2, is_train=True):
    """
    Create the generator network
    :param z: Input z
    :param out_channel_dim: The number of channels in the output image
    :param is_train: Boolean if generator is being used for training
    :return: The tensor output of the generator
    """
    # TODO: Implement Function

    with tf.variable_scope("generator", reuse=reuse):
        
        # connect random input z to a dense layer
        x1 = tf.layers.dense(inputs=z, units=2*2*512, activation=None, name="x1.dense")

        # apply batch normalization and leaky relu activation
        x2 = tf.reshape(x1, shape=(-1, 2, 2, 512))
        x2 = tf.layers.batch_normalization(inputs=x2, training=is_train, name="x2.bn")
        x2 = tf.maximum(alpha * x2, x2, name="x2.leaky_relu")

        # input: 2*2*512 
        x3 = tf.layers.conv2d_transpose(inputs=x2, filters=256, activation=None, kernel_size=5, strides=2, padding='valid', name="x3.conv2d_t")
        x3 = tf.layers.batch_normalization(inputs=x3, training=is_train, name="x3.bn")
        x3 = tf.maximum(alpha * x3, x3, name="x3.leaky_relu")

        # input: 7 * 7 * 256 
        x4 = tf.layers.conv2d_transpose(inputs=x3, filters=128, activation=None, kernel_size=5, strides=2, padding='same', name="x4.conv2d_t")
        x4 = tf.layers.batch_normalization(inputs=x4, training=is_train, name="x4.bn")
        x4 = tf.maximum(alpha * x4, x4, name="x4.leaky_relu")

        # input: 14 * 14 * 128 
        x5 = tf.layers.conv2d_transpose(inputs=x4, filters=out_channel_dim, activation=None, kernel_size=5, strides=2, padding='same', name="x5.conv2d_t")
        x5 = tf.layers.batch_normalization(inputs=x5, training=is_train, name="x5.bn")
        x5 = tf.maximum(alpha * x5, x5, name="x5.leaky_relu")

        # rescale output using tanh
        g_model = tf.tanh(x5, name="x5.tanh")
        
        return g_model


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
#tests.test_generator(generator, tf)


# ### Loss
# Implement `model_loss` to build the GANs for training and calculate the loss.  The function should return a tuple of (discriminator loss, generator loss).  Use the following functions you implemented:
# - `discriminator(images, reuse=False)`
# - `generator(z, out_channel_dim, is_train=True)`

# In[22]:

def model_loss(input_real, input_z, out_channel_dim, smooth=0.1):
    """
    Get the loss for the discriminator and generator
    :param input_real: Images from the real dataset
    :param input_z: Z input
    :param out_channel_dim: The number of channels in the output image
    :return: A tuple of (discriminator loss, generator loss)
    """
    # TODO: Implement Function
    
    g_model = generator(z = input_z, out_channel_dim=out_channel_dim)
    
    d_model_real, d_logits_real = discriminator(images=input_real)
    d_model_fake, d_logits_fake = discriminator(images=g_model, reuse=True)
    
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_logits_fake)))
    
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_logits_real) * (1. - smooth) ))
    
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_logits_fake)))
    
    d_loss = d_loss_real + d_loss_fake
    return d_loss, g_loss


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
#tests.test_model_loss(model_loss)


# ### Optimization
# Implement `model_opt` to create the optimization operations for the GANs. Use [`tf.trainable_variables`](https://www.tensorflow.org/api_docs/python/tf/trainable_variables) to get all the trainable variables.  Filter the variables with names that are in the discriminator and generator scope names.  The function should return a tuple of (discriminator training operation, generator training operation).

# In[23]:

def model_opt(d_loss, g_loss, learning_rate, beta1):
    """
    Get optimization operations
    :param d_loss: Discriminator loss Tensor
    :param g_loss: Generator loss Tensor
    :param learning_rate: Learning Rate Placeholder
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :return: A tuple of (discriminator training operation, generator training operation)
    """

    all_vars = [v for v in tf.trainable_variables() ]
    print("all vars found: ",len(all_vars))
    
    g_vars = [v for v in all_vars if v.name.startswith('generator')]
    print("generator vars found: ",len(g_vars), ": ", [g.name for g in g_vars])
    
    d_vars = [v for v in all_vars if v.name.startswith('discriminator')]
    print("discriminator vars found: ", len(d_vars), ": ", [d.name for d in d_vars])
    
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)
        d_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
        
        return d_opt, g_opt


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
#tests.test_model_opt(model_opt, tf)


# ## Neural Network Training
# ### Show Output
# Use this function to show the current output of the generator during training. It will help you determine how well the GANs is training.

# In[24]:

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np

def show_generator_output(sess, n_images, input_z, out_channel_dim, image_mode):
    """
    Show example output for the generator
    :param sess: TensorFlow session
    :param n_images: Number of Images to display
    :param input_z: Input Z Tensor
    :param out_channel_dim: The number of channels in the output image
    :param image_mode: The mode to use for images ("RGB" or "L")
    """
    cmap = None if image_mode == 'RGB' else 'gray'
    z_dim = input_z.get_shape().as_list()[-1]
    example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])

    samples = sess.run(
        generator(input_z, out_channel_dim, reuse=True, is_train=False),
        feed_dict={input_z: example_z})

    images_grid = helper.images_square_grid(samples, image_mode)
    #pyplot.imshow(images_grid, cmap=cmap)
    #pyplot.show()


# ### Train
# Implement `train` to build and train the GANs.  Use the following functions you implemented:
# - `model_inputs(image_width, image_height, image_channels, z_dim)`
# - `model_loss(input_real, input_z, out_channel_dim)`
# - `model_opt(d_loss, g_loss, learning_rate, beta1)`
# 
# Use the `show_generator_output` to show `generator` output while you train. Running `show_generator_output` for every batch will drastically increase training time and increase the size of the notebook.  It's recommended to print the `generator` output every 100 batches.

# In[25]:

def train(epoch_count, batch_size, z_dim, learning_rate, beta1, get_batches, data_shape, data_image_mode):
    """
    Train the GAN
    :param epoch_count: Number of epochs
    :param batch_size: Batch Size
    :param z_dim: Z dimension
    :param learning_rate: Learning Rate
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :param get_batches: Function to get batches
    :param data_shape: Shape of the data
    :param data_image_mode: The image mode to use for images ("RGB" or "L")
    """


    input_real, input_z, learn_rate = model_inputs(data_shape[1], data_shape[2], data_shape[3], z_dim)    
    d_loss, g_loss = model_loss(input_real=input_real, input_z=input_z, out_channel_dim=data_shape[3])    
    d_opt, g_opt = model_opt(d_loss=d_loss, g_loss=g_loss, beta1=beta1, learning_rate=learning_rate)
    
    show_fig_every_batch = 100
    show_loss_every_batch = 50
    
    num_channels = data_shape[3]
    if(num_channels==3):
        image_mode_code = "RGB"
    else:
        image_mode_code = "L"
        
    show_images = 16
    random_constant_z = tf.random_uniform(dtype=tf.float32, minval=-1, maxval=1, shape=(show_images,z_dim))
    
    with tf.Session() as sess:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        sess.run(tf.global_variables_initializer())
        for epoch_i in range(epoch_count):

            batch_num = 0
            for batch_images in get_batches(batch_size):
                
                batch_num +=1
                sample_z = np.random.uniform(high=1, low=-1, size=[batch_size, z_dim])                
                
                _ , discr_loss = sess.run([d_opt, d_loss], feed_dict={input_real: batch_images, input_z: sample_z})
                
                _ , gener_loss = sess.run([g_opt, g_loss], feed_dict={input_real: batch_images, input_z: sample_z}) 

                if(batch_num%show_loss_every_batch == 0 ):
                    print("discriminator loss: ", discr_loss, "generator loss: ", gener_loss)
                
                if(batch_num%show_fig_every_batch == 0):
                    show_generator_output(sess=sess, n_images=show_images, input_z=random_constant_z, out_channel_dim=num_channels, image_mode=image_mode_code)
                


# ### MNIST
# Test your GANs architecture on MNIST.  After 2 epochs, the GANs should be able to generate images that look like handwritten digits.  Make sure the loss of the generator is lower than the loss of the discriminator or close to 0.

# In[28]:

batch_size = 64
z_dim = 250
learning_rate = 0.0002
beta1 = 0.4


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
epochs = 1

mnist_dataset = helper.Dataset('mnist', glob(os.path.join(data_dir, 'mnist/*.jpg')))


# In[ ]:

with tf.Graph().as_default():
    train(epochs, batch_size, z_dim, learning_rate, beta1, mnist_dataset.get_batches,
          mnist_dataset.shape, mnist_dataset.image_mode)


# ### CelebA
# Run your GANs on CelebA.  It will take around 20 minutes on the average GPU to run one epoch.  You can run the whole epoch or stop when it starts to generate realistic faces.

# In[ ]:

#batch_size = 64
#z_dim = 100
#learning_rate = 0.0003
#beta1 = 0.3
#
#
#"""
#DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
#"""
#epochs = 1

#celeba_dataset = helper.Dataset('celeba', glob(os.path.join(data_dir, 'img_align_celeba/*.jpg')))
#with tf.Graph().as_default():
#    train(epochs, batch_size, z_dim, learning_rate, beta1, celeba_dataset.get_batches,
#          celeba_dataset.shape, celeba_dataset.image_mode)


# ### Submitting This Project
# When submitting this project, make sure to run all the cells before saving the notebook. Save the notebook file as "dlnd_face_generation.ipynb" and save it as a HTML file under "File" -> "Download as". Include the "helper.py" and "problem_unittests.py" files in your submission.

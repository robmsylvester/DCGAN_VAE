import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import l2_regularizer, xavier_initializer, fully_connected, flatten

"""
In this file, there are higher-level abstractions for different layers
that will be used in the Generative Adversarial Network as well as the
Variational Autoencoder. Since the architecture of these networks is still
a very active research topic, we will define many activation functions
and a few others that might not be necessarily common in the literature

Note: All convolutional operations expect NHWC format"""

def binary_cross_entropy(logits, labels, epsilon=1e-7, name="binary_cross_entropy"):
    with tf.variable_scope(name):
        
        clipped_flat_logits = tf.clip_by_value(flatten(logits), epsilon, 1 - epsilon) #protection against zero
        flat_labels = flatten(labels)
        
        return -tf.reduce_sum(flat_labels * tf.log(clipped_flat_logits) +\
                              (1-flat_labels) * tf.log(1 - clipped_flat_logits), 1)
    
def univariate_gaussian_kl_divergence(mu_1, sigma_1, mu_2, sigma_2, name="gaussian_KL"):
    """Given two univariate probability distributions with variances mu and sigma, calculuates relative entropy
    between them, given that these two distributions are defined by their means and standard deviations.
    
    Formally, this calculates KL(p,q) =  -integral[ p(x) log (q(x) ] dx + integral[ p(x) log (p(x)) ] dx
     where p is defined by mu_1, mu_2, and q is defined by mu_2, sigma_2
     
     Returns a summation over all dimensions over the size of the 1_d argument tensors
     
     (standard deviations should be positive...so let's keep them that way with the absolute values)
    
    For a decent mathematical breakdown, see this:
    https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    """

    with tf.variable_scope(name):
        return tf.reduce_sum( ( tf.log(tf.abs(sigma_2) / tf.abs(sigma_1)) +\
                 ( tf.square(sigma_1) + tf.square(mu_1 - mu_2) ) / (2*tf.square(sigma_2)) - 0.5), 1)
        
        

def univariate_normal_gaussian_kl_divergence(mu, sigma, name="gaussian_normal_KL"):
    """The relative entropy in this case is against a normal distribution, that is to say, in the above
    formula for the univariate_gaussian_kl_divergence, mu_2=0 and sigma_2=1. This makes the formula 
    a bit easier and faster on a computer by eliminating some variables because one of the gaussians
    params are fixed
    
    Another thing really nice happens here too, and that is that the sigmas that we generated from the
    encoder...you know the ones that could be negative numbers...disappear with that tf.square function.
    """
    with tf.variable_scope(name):
        return 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(tf.square(sigma)) - 1, 1)

#various implementations of this exist, particularly in how the negatives are calculated
#for more, read https://arxiv.org/abs/1502.01852
def parametric_relu(X, regularizer=None, name="parametric_relu"):
    with tf.variable_scope(name):
        alphas = tf.get_variable('alphas',
                                 regularizer=regularizer,
                                 dtype=X.dtype,
                                 shape=X.get_shape().as_list()[-1], 
                                 initializer=tf.constant_initializer(0.01))
        positives = tf.nn.relu(X)
        negatives = alphas*(tf.subtract(X, tf.abs(X))) * 0.5
        return positives + negatives

#this implementation assumes alpha will be less than 1. It would be stupid if it weren't.
def leaky_relu(X, alpha=0.2):
    return tf.maximum(X, alpha*X)

#https://arxiv.org/abs/1302.4389
#This doesn't need a scope. There aren't trainable parameters here. It's just a pool
def maxout(X, num_maxout_units, axis=None):
    input_shape = X.get_shape().as_list()
    
    axis = -1 if axis is None else axis
    
    num_filters = input_shape[axis]
    
    if num_filters % num_maxout_units != 0:
        raise ValueError, "num filters (%d) must be divisible by num maxout units (%d)" % (num_filters, num_maxout_units)
    
    output_shape = input_shape.copy()
    output_shape[axis] = num_maxout_units
    output_shape += [num_filters // num_maxout_units]
    return tf.reduce_max(tf.reshape(X, output_shape), -1, keep_dims=False)


def conv(X,
         output_filter_size,
         kernel=[5,5],
         strides=[2,2],
         w_initializer=xavier_initializer(),
         regularizer=None,
         name="conv"):
    
    with tf.variable_scope(name):
        W = tf.get_variable('W_conv',
                            regularizer=regularizer,
                            dtype=X.dtype,
                            shape=[kernel[0], kernel[1], X.get_shape().as_list()[-1], output_filter_size],
                            initializer=w_initializer)
        b = tf.get_variable('b_conv',
                            regularizer=regularizer,
                            dtype=X.dtype,
                            shape=[output_filter_size],
                            initializer=tf.zeros_initializer(dtype=X.dtype))
        
        return tf.nn.bias_add( tf.nn.conv2d(X,
                                            W,
                                            strides=[1,strides[0],strides[1],1],
                                            padding='SAME',
                                            name="conv2d"), b)

def conv_transpose(X,
                 output_shape,
                 kernel=[5,5],
                 strides=[2,2],
                 w_initializer=xavier_initializer(),
                 regularizer=None,
                 name="conv_t"):
       
    with tf.variable_scope(name):
        W = tf.get_variable('W_conv_T',
                            regularizer=regularizer,
                            dtype=X.dtype,
                            shape=[kernel[0], kernel[1], output_shape[-1], X.get_shape().as_list()[-1]],
                            initializer=w_initializer)
        b = tf.get_variable('b_conv_T',
                            regularizer=regularizer,
                            dtype=X.dtype,
                            shape=[output_shape[-1]],
                            initializer=tf.zeros_initializer(dtype=X.dtype))
        X_shape = X.get_shape().as_list()
                
        return tf.nn.bias_add( tf.nn.conv2d_transpose(X,
                                            W,
                                            output_shape,
                                            strides=[1,strides[0],strides[1],1],
                                            padding='SAME',
                                            name="conv2d_T"), b)

#get the convolutional output size given same padding and equal strides
def conv_out_size(sz, stride):
    return int(np.ceil(float(sz) / float(stride)))


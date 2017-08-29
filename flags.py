import tensorflow as tf
import os
import sys

#==========================Dataset Parameters====================================
tf.app.flags.DEFINE_string("dataset", 'mnist',
                          "Supported datasets for inception experiments. If using custom images, enter None")

#==========================Model Parameters==================================
tf.app.flags.DEFINE_integer("batch_size", 128,
                            "The number of images to feed per training step.")


#=========================Learning Rate======================================
#tf.app.flags.DEFINE_float("init_learning_rate", 0.01,
#                          "The initial learning rate before any decays")
#tf.app.flags.DEFINE_float("learning_rate_decay", 0.999,
#                          "When the learning rate decays, it will be multiplied by this much")
#tf.app.flags.DEFINE_float("min_learning_rate", 0.0001,
#                          "The learning rate will never fall below this number. Decays resulting in less will be ignored")
#tf.app.flags.DEFINE_integer("decay_steps", 100,
#                          "The number of steps before learning rate is multiplied by learning rate decay")



#==========================Regularization Parameters==================================
tf.app.flags.DEFINE_float("l2_lambda_weight", 0.001,
                          "The initial weight to use as a multiplier to the L2 sum. This can be decayed over training.")
                          
tf.app.flags.DEFINE_float("l2_lambda_weight_decay", 0.999,
                          "Every decay_steps number of training steps, the current lambda weight will be multiplied by this number.")
tf.app.flags.DEFINE_integer("l2_lambda_weight_decay_steps", 1000,
                          "The number of weight decay steps before running a lambda weight decay operation.")

def flag_test():
    supported_datasets = ['cifar100', 'mnist', None]
    assert tf.app.flags.FLAGS.dataset in supported_datasets, "Your dataset must be cifar100, mnist, or None"
                          
    assert tf.app.flags.FLAGS.batch_size > 0, "Batch size must be a positive integer"
    
    assert tf.app.flags.FLAGS.l2_lambda_weight > 0, "l2 lambda weight must be positive"
    assert tf.app.flags.FLAGS.l2_lambda_weight_decay > 0. and tf.app.flags.FLAGS.l2_lambda_weight_decay <= 1., "Lambda weight decay for L2 must be within the range (0, 1.]"
    assert tf.app.flags.FLAGS.l2_lambda_weight_decay_steps > 0, "l2 lambda weight decay steps must be positive"
                         
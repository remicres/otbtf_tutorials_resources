from tricks import *
import sys
import os

nclasses=6

def myModel(x1,x2):
  
  # The 20m spacing branch (input patches: 8x8x3)
  conv1_x1 = tf.layers.conv2d(inputs=x1, filters=16, kernel_size=[3,3], padding="valid",
                              activation=tf.nn.relu) # out size: 6x6x16
  conv2_x1 = tf.layers.conv2d(inputs=conv1_x1, filters=32, kernel_size=[3,3], padding="valid",
                              activation=tf.nn.relu) # out size: 4x4x32
  conv3_x1 = tf.layers.conv2d(inputs=conv2_x1, filters=64, kernel_size=[3,3], padding="valid",
                              activation=tf.nn.relu) # out size: 2x2x64
  conv4_x1 = tf.layers.conv2d(inputs=conv3_x1, filters=64, kernel_size=[2,2], padding="valid",
                              activation=tf.nn.relu) # out size: 1x1x64
  
  # The 10m spacing branch (input patches: 16x16x4)
  conv1_x2 = tf.layers.conv2d(inputs=x2, filters=16, kernel_size=[5,5], padding="valid",
                              activation=tf.nn.relu) # out size: 12x12x16
  conv2_x2 = tf.layers.conv2d(inputs=conv1_x2, filters=32, kernel_size=[3,3], padding="valid",
                              activation=tf.nn.relu) # out size: 10x10x32
  conv3_x2 = tf.layers.conv2d(inputs=conv2_x2, filters=64, kernel_size=[3,3], padding="valid",
                              activation=tf.nn.relu) # out size: 8x8x64
  conv4_x2 = tf.layers.conv2d(inputs=conv3_x2, filters=64, kernel_size=[3,3], padding="valid",
                              activation=tf.nn.relu) # out size: 6x6x64
  conv5_x2 = tf.layers.conv2d(inputs=conv4_x2, filters=64, kernel_size=[3,3], padding="valid",
                              activation=tf.nn.relu) # out size: 4x4x64
  pool1_x2 = tf.layers.max_pooling2d(inputs=conv5_x2, pool_size=[2, 2],
                              strides=2) # out size: 2x2x64
  conv6_x2 = tf.layers.conv2d(inputs=pool1_x2, filters=64, kernel_size=[2,2], padding="valid",
                              activation=tf.nn.relu) # out size: 1x1x64
  
  # Stack features from the two branches
  features = tf.reshape(tf.stack([conv4_x1, conv6_x2], axis=3), 
                        shape=[-1, 128], name="features")
  
  # Fully connected layer
  dense_1 = tf.layers.dense(inputs=features, units=128, activation=tf.nn.relu)

  # Neurons for classes
  estimated = tf.layers.dense(inputs=dense_1, units=nclasses, activation=None)
  estimated_label = tf.argmax(estimated, 1, name="prediction")
  
  return estimated, estimated_label
 
""" Main """
# check number of arguments
if len(sys.argv) != 2:
  print("Usage : <output directory for SavedModel>")
  sys.exit(1)

# Create the graph
with tf.Graph().as_default():
  
  # Placeholders
  x1 = tf.placeholder(tf.float32, [None, None, None, 6], name="x1")
  x2 = tf.placeholder(tf.float32, [None, None, None, 4], name="x2")
  y  = tf.placeholder(tf.int32  , [None, None, None, 1], name="y")
  lr = tf.placeholder_with_default(tf.constant(0.0002, dtype=tf.float32, shape=[]), 
                                   shape=[], name="lr")
  
  # Output
  y_estimated, y_label = myModel(x1,x2)
  
  # Loss function
  cost = tf.losses.sparse_softmax_cross_entropy(labels=tf.reshape(y, [-1, 1]), 
                                                logits=tf.reshape(y_estimated, [-1, nclasses]))
  
  # Optimizer
  optimizer = tf.train.AdamOptimizer(learning_rate=lr, name="optimizer").minimize(cost)
  
  # Initializer, saver, session
  init = tf.global_variables_initializer()
  saver = tf.train.Saver( max_to_keep=20 )
  sess = tf.Session()
  sess.run(init)

  # Create a SavedModel
  CreateSavedModel(sess, ["x1:0", "x2:0", "y:0"], ["features:0", "prediction:0"], sys.argv[1])

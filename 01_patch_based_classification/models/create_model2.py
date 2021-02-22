from tricks import *
import sys
import os

nclasses=6

def myModel(x):
  
  # input patches: 16x16x4
  conv1 = tf.layers.conv2d(inputs=x, filters=16, kernel_size=[5,5], padding="valid", 
                           activation=tf.nn.relu) # out size: 12x12x16
  conv2 = tf.layers.conv2d(inputs=conv1, filters=16, kernel_size=[3,3], padding="valid", 
                           activation=tf.nn.relu) # out size: 10x10x16
  conv3 = tf.layers.conv2d(inputs=conv2, filters=16, kernel_size=[3,3], padding="valid", 
                           activation=tf.nn.relu) # out size: 8x8x16
  conv4 = tf.layers.conv2d(inputs=conv3, filters=32, kernel_size=[3,3], padding="valid",
                           activation=tf.nn.relu) # out size: 6x6x32
  conv5 = tf.layers.conv2d(inputs=conv4, filters=32, kernel_size=[3,3], padding="valid",
                           activation=tf.nn.relu) # out size: 4x4x32
  conv6 = tf.layers.conv2d(inputs=conv5, filters=32, kernel_size=[3,3], padding="valid",
                           activation=tf.nn.relu) # out size: 2x2x32
  conv7 = tf.layers.conv2d(inputs=conv6, filters=32, kernel_size=[2,2], padding="valid",
                           activation=tf.nn.relu) # out size: 1x1x32
  
  # Features
  features = tf.reshape(conv7, shape=[-1, 32], name="features")
  
  # Neurons for classes
  estimated = tf.layers.dense(inputs=features, units=nclasses, activation=None)
  estimated_label = tf.argmax(estimated, 1, name="prediction")

  return estimated, estimated_label

""" Main """
if len(sys.argv) != 2:
  print("Usage : <output directory for SavedModel>")
  sys.exit(1)

# Create the TensorFlow graph
with tf.Graph().as_default():
  
  # Placeholders
  x = tf.placeholder(tf.float32, [None, None, None, 4], name="x")
  y = tf.placeholder(tf.int32  , [None, None, None, 1], name="y")
  lr = tf.placeholder_with_default(tf.constant(0.0002, dtype=tf.float32, shape=[]),
                                   shape=[], name="lr")
  
  # Output
  y_estimated, y_label = myModel(x)
  
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
  create_savedmodel(sess, ["x:0", "y:0"], ["features:0", "prediction:0"], sys.argv[1])

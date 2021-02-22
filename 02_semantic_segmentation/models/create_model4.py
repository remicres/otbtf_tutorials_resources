from tricks import *
import sys
import os

nclasses=2

# Convolution block with strides 2 ("downsampling")
def _conv(inp, n, k_size=3, strides=2, activ=tf.nn.relu):
  out = tf.layers.conv2d(
    inputs=inp, 
    filters=n, 
    kernel_size=[k_size, k_size], 
    padding="same", 
    strides=(strides, strides),
    activation=activ)
  return out

# Transposed convolution block with strides 2 ("upsampling")
def _dconv(inp, n, k_size=3, strides=2, activ=tf.nn.relu):
  out =  tf.layers.conv2d_transpose(
    inputs=inp,
    filters=n,
    strides=(strides,strides),
    kernel_size=[k_size, k_size],
    padding="same",
    activation=activ)
  return out

def myModel(x):

  depth = 16
  
  # Encoding
  conv1   = _conv(x,        1*depth)         #  64 x 64 --> 32 x 32 (31 x 31)
  conv2   = _conv(conv1,    2*depth)         #  32 x 32 --> 16 x 16 (15 x 15)
  conv3   = _conv(conv2,    4*depth)         #  16 x 16 -->  8 x  8 ( 7 x  7)
  conv4   = _conv(conv3,    4*depth)         #   8 x  8 -->  4 x  4 ( 3 x  3)
  
  # Decoding (with skip connections)
  deconv1 = _dconv(conv4,           4*depth) #  4  x  4 -->  8 x  8 ( 5 x  5)
  deconv2 = _dconv(deconv1 + conv3, 2*depth) #  8  x  8 --> 16 x 16 ( 9 x  9)
  deconv3 = _dconv(deconv2 + conv2, 1*depth) # 16  x 16 --> 32 x 32 (17 x 17)
  deconv4 = _dconv(deconv3 + conv1, 1*depth) # 32  x 32 --> 64 x 64 (33 x 33)
  
  # Neurons for classes
  estimated = tf.layers.dense(inputs=deconv4, units=nclasses, activation=None)
  
  return estimated
 
""" Main """
# check number of arguments
if len(sys.argv) != 2:
  print("Usage : <output directory for SavedModel>")
  sys.exit(1)

# Create the graph
with tf.Graph().as_default():
  
  # Placeholders
  x = tf.placeholder(tf.float32, [None, None, None, 4], name="x")
  y  = tf.placeholder(tf.int32 , [None, None, None, 1], name="y")
  lr = tf.placeholder_with_default(tf.constant(0.0002, dtype=tf.float32, shape=[]), 
              shape=[], name="lr")
  
  # Output neurons of the model
  y_estimated = myModel(x)

  # Prediction output
  y_class = tf.argmax(y_estimated, axis=3)

  # Add the 4th dimension to have (#, :, :, 1) so the OTBTF apps
  # know that the output image has 1 component
  y_pred = tf.expand_dims(y_class, axis=-1, name="prediction")

  # Prediction output (FCN)
  y_pred_fcn = tf.identity(y_pred[:, 16:-16, 16:-16, :], name="prediction_fcn")
  
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
  create_savedmodel(sess, ["x:0","y:0"], ["prediction:0", "prediction_fcn:0"], sys.argv[1])

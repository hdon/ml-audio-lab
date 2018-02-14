import tensorflow as tf
from util import iterWindow

class Model:
  def __init__(self
  , num_features
  , num_labels
  , lr_start = 0.01
  , lr_end = 0.001
  , lr_decay_steps = 1600
  , lr = None
  , num_hidden = [32]
  , activation = [tf.tanh, tf.tanh]
  , stddev = 0.1
  , auto_encoder = True
  ):
    # Validate arguments
    if len(num_hidden) != len(activation)-1:
      raise IndexError('len(num_hidden) != len(activation)-1')

    # Keep track of how many training steps we've taken
    global_step = tf.Variable(0, trainable=False)

    # If no learning rate is supplied, assume a polynomial_decay
    if lr is None:
      lr = tf.train.polynomial_decay(
        lr_start
      , global_step
      , lr_decay_steps
      , lr_end
      , power=0.5
      )

    x = tf.placeholder(tf.float32, [None, num_features], name="x")
    if auto_encoder:
      y = x
    else:
      y = tf.placeholder(tf.float32, [None, num_labels], name="y")

    w = []
    b = []
    a = [x]

    # build weights
    for dims in iterWindow([num_features] + num_hidden + [num_labels], 2):
      w.append(tf.Variable(tf.truncated_normal(dims, stddev=stddev)))
    # build biases
    for dims in num_hidden + [num_labels]:
      b.append(tf.Variable(tf.truncated_normal([dims], stddev=stddev)))
    # build network layers
    for iLayer, (dims, activate) in enumerate(zip(num_hidden + [num_labels], activation)):
      a.append(activate(
        tf.add(tf.matmul(a[iLayer], w[iLayer]), b[iLayer])
      ))

    estimated_y = a[-1]
    cost_per = tf.pow(y - estimated_y, 2)
    cost = tf.reduce_mean(cost_per)

    self.x = x
    self.y = y
    self.w = w
    self.b = b
    self.a = a
    self.estimated_y = estimated_y
    self.cost_per = cost_per
    self.cost = cost
    self.global_step = global_step

    self.num_features = num_features
    self.num_labels = num_labels
    self.lr_start = lr_start
    self.lr_end = lr_end
    self.lr_decay_steps = lr_decay_steps
    self.lr = lr
    self.stddev = stddev
    self.num_hidden = num_hidden
    self.activation = activation
    self.auto_encoder = auto_encoder

  def getModelFilename(self):
    return '%(num_hidden)s-%(stddev)s-%(activation)s-lr-%(lr_start)s-%(lr_end)s-%(lr_decay_steps)s' % {
      'num_hidden': ','.join(map(str, self.num_hidden))
    , 'activation': ','.join(map(lambda f:f.__name__, self.activation))
    , 'stddev': self.stddev
    , 'lr_start': self.lr_start
    , 'lr_end': self.lr_end
    , 'lr_decay_steps': self.lr_decay_steps
    }

    #for n in num_hidden:
    #w1 = tf.Variable(tf.truncated_normal([num_features, num_hidden], stddev=0.1))
    #w2 = tf.Variable(tf.truncated_normal([num_hidden, num_hidden], stddev=0.1))
    #w3 = tf.Variable(tf.truncated_normal([num_hidden, num_features], stddev=0.1))
    #b1 = tf.Variable(tf.truncated_normal([num_hidden], stddev=0.1))
    #b2 = tf.Variable(tf.truncated_normal([num_hidden], stddev=0.1))
    #b3 = tf.Variable(tf.truncated_normal([num_labels], stddev=0.1))
#
    #layer_1 = tf.nn.relu(tf.add(tf.matmul(x, w1), b1))
    #layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, w2), b2))
    #layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, w3), b3))
#
    #self.estimated_y = layer_3
    #self.y = x # autoencoder just fits output to input
    #self.cost_per = tf.pow(y - estimated_y, 2)
    #self.cost = tf.reduce_mean(cost_per)

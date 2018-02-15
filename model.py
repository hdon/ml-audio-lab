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

    self.x = x
    self.y = y
    self.w = w
    self.b = b
    self.a = a
    self.estimated_y = estimated_y
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
    lr = self.lr if self.lr is not None else '%s-%s-%s' % (lr_start, lr_end, lr_decay_steps)
    return '%(num_hidden)s-%(stddev)s-%(activation)s-lr-%(lr)s' % {
      'num_hidden': ','.join(map(str, self.num_hidden))
    , 'activation': ','.join(map(lambda f:f.__name__, self.activation))
    , 'stddev': self.stddev
    , 'lr': lr
    }

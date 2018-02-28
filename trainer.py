import tensorflow as tf
import os, code

optimizers_by_name = {
  'adam': tf.train.AdamOptimizer
, 'gd': tf.train.GradientDescentOptimizer
}

# TODO summaries with same parameters are getting grouped together in
# tensorboard -- probably need to disambiguate them further. probably
# should check the disk for what files already exist and then bump a
# counter until a unique name is found
class Trainer:
  def __init__(self
  , model
  , x
  , y
  , load_checkpoint_filename=None
  , target_steps = 1600
  , target_cost = 0.000001
  , steps_per_summary = 10
  , y_morpher = None
  , optimizer='adam'
  , cost = 'mse'
  ):
    self.model = model
    self.x = x
    self.y = y # TODO feed this in too if (x is not y)

    self.y_morpher = y_morpher
    estimated_y = model.estimated_y
    if y_morpher is not None:
      self.morphed_y = y_morpher(model.estimated_y)
      estimated_y = self.morphed_y

    if cost == 'mse':
      self.cost_per = tf.pow(model.y - estimated_y, 2)
      self.cost = tf.reduce_mean(self.cost_per)
    elif cost == 'ces':
      self.cost_per = - model.y * tf.log(model.estimated_y + 1.0001)
      self.cost = tf.reduce_sum(self.cost_per)
    else:
      raise ValueError('unknown cost function')

    self.load_checkpoint_filename = load_checkpoint_filename
    self.target_steps = target_steps
    self.target_cost = target_cost
    self.steps_per_summary = steps_per_summary
    self.learning_rate = model.lr
    if not callable(optimizer):
      if type(optimizer) is not str:
        raise ValueError('invalid optimizer')
      self.optimizer_name = optimizer
      optimizer = optimizers_by_name[optimizer]
    else:
      self.optimizer_name = optimizer.__name__
    self.optimizer = optimizer(self.learning_rate) \
      .minimize(self.cost, global_step=model.global_step)
    tf.summary.scalar('cost', self.cost)
    self.summary_op = tf.summary.merge_all()
    self.log_dir_name = None
    self.writer = tf.summary.FileWriter(self.getLogDirName(), graph=tf.get_default_graph())


  def getCheckpointFilename(self):
    return '%(model)s-t-%(step)s' % {
      'model': self.model.getModelFilename()
    , 'step': self.steps_trained
    }

  def getLogDirName(self):
    if self.log_dir_name is None:
      log_dir_name_part = 'log/%(model)s-t-%(step)s-r-%%02d' % {
        'model': self.model.getModelFilename()
      , 'step': self.target_steps
      }
      n = 0
      while 1:
        log_dir_name = log_dir_name_part % n
        if not os.path.exists(log_dir_name):
          # use previous log file if we were asked to load a previously
          # saved checkpoint; TODO more robust system for continuing
          # training
          if self.load_checkpoint_filename is not None and n > 0:
            n -= 1
          self.log_dir_name = log_dir_name
          break
        n += 1
    return self.log_dir_name

  def train(self):
    self.keep_training = True
    x = self.x

    # How many steps trained so far?
    self.steps_trained = 0

    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1)) as sess:
      saver = tf.train.Saver()
      init = tf.global_variables_initializer()
      sess.run(init)
      if self.load_checkpoint_filename:
        saver.restore(sess, self.load_checkpoint_filename)
        print('model restored')
        self.steps_trained = sess.run(self.model.global_step)
        # TODO restore epochs counter!

      while self.keep_training:
        try:
          # 
          feed_dict = {
            self.model.x: self.x
          }
          if not self.model.auto_encoder:
            feed_dict[self.model.y] = self.y
          sess.run(self.optimizer, feed_dict=feed_dict)
          if self.steps_trained % self.steps_per_summary == 0:
            training_cost, summary = sess.run(
              [self.cost, self.summary_op]
            , feed_dict=feed_dict
            )
            print('% 7d/% 7d cost=%.9f' % (
              self.steps_trained
            , self.target_steps
            , training_cost
            ))
            self.writer.add_summary(summary, self.steps_trained)

            if training_cost <= self.target_cost:
              print('target cost achieved, no more training')
              break
          self.steps_trained += 1
          if self.steps_trained > self.target_steps:
            self.keep_training = False
        except KeyboardInterrupt as e:
          print()
          print()
          print('entering interactive shell')
          print('REMEMBER not to raise SYSTEMEXIT if you want your model saved!!!')
          print('to save and quit, assign self.keep_training=False')
          print()
          print()
          code.interact(local=locals())

      # save checkpoint
      checkpoint_filename = self.getCheckpointFilename()
      saver.save(sess, 'checkpoints/' + checkpoint_filename)
      print('model saved')

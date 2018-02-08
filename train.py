import numpy as np
import tensorflow as tf
import scipy.signal
import scipy.io.wavfile
import code, sys, os

INFINITY = float('inf')

input_filename = sys.argv[1]
load_checkpoint_filename = sys.argv[2] if len(sys.argv) > 2 else None

print 'reading', input_filename

def process_input(input_filename, max_samples = INFINITY):
  sample_rate, input_samples = scipy.io.wavfile.read(input_filename, mmap=True)
  if len(input_samples) > max_samples:
    input_samples = input_samples[0:max_samples]
  print input_samples.shape
  # collect some diagnostic info about samples
  print 'input samples mean:', np.mean(input_samples)
  print 'input samples std:', np.std(input_samples)
  # for our purposes we can ignore dft_freqs and dft_times
  print 'stft'
  stft_freqs, stft_times, stft = scipy.signal.stft(input_samples)
  return sample_rate, stft_freqs, stft_times, stft

def foo(output_filename):
  'inverse an stft and write to wav file'
  # we can ignore istft_times
  print 'istft'
  istft_times, output_samples = scipy.signal.istft(stft)
  #print 'converting to int'
  #output_samples = output_samples.astype(int)
  # collect some diagnostic info about samples
  print 'output samples mean:', np.mean(output_samples)
  print 'output samples std:', np.std(output_samples)
  #scipy.io.wavfile.write('foo.wav', 8000, output_samples)
  print 'quieting'
  d = np.max(output_samples) - np.min(output_samples)
  m = np.max(output_samples) - d/2
  output_samples = (output_samples - m) / d
  print 'output samples mean:', np.mean(output_samples)
  print 'output samples std:', np.std(output_samples)
  print 'writing'
  scipy.io.wavfile.write(output_filename, 8000, output_samples)
  #output_samples.tofile('foo.pcm')

  print 'bye'

try:
  input_tracks = []
  sample_rate, stft_freqs, stft_times, stft = process_input(input_filename, 80000)
  print stft_freqs.shape
  print stft_times.shape
  print 'stft.shape=', stft.shape

  # Hyperparameters
  learning_rate = 0.1
  training_epochs = 200000
  target_cost = 0.01
  num_hidden = 64
  # training state
  display_step = 10
  n_samples = stft.shape[0]
  keep_training = True
  epochs = 0

  num_features = stft.shape[0]
  X = tf.placeholder(tf.float32, [None, num_features], name="X")
  w1 = tf.Variable(tf.random_normal([num_features, num_hidden]))
  w2 = tf.Variable(tf.random_normal([num_hidden, num_features]))
  b1 = tf.Variable(tf.random_normal([num_hidden]))
  b2 = tf.Variable(tf.random_normal([num_features]))

  layer_1 = tf.tanh(tf.add(tf.matmul(X, w1), b1))
  layer_2 = tf.tanh(tf.add(tf.matmul(layer_1, w2), b2))

  y = layer_2
  y_ = X # autoencoder just fits output to input
  cost = tf.reduce_mean(tf.pow(y_ - y, 2))
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

  # summaries for tensorboard
  tf.summary.scalar('cost', cost)
  summary_op = tf.summary.merge_all()
  # writer for summaries
  writer = tf.summary.FileWriter('log', graph=tf.get_default_graph())
  # transposing this here seems kinda silly? maybe i should do it earlier?
  # or maybe there is a more tensorflowy way to do this?
  # remove imaginary components, transpose, and scale to -1,1 range
  stft_prepared = np.transpose(np.real(stft))
  low = np.min(stft_prepared)
  high = np.max(stft_prepared)
  stft_prepared = (stft_prepared - (low+high)/2.0) / ((high-low)/2.0)

  saver = tf.train.Saver()

  with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1)) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    if load_checkpoint_filename:
      saver.restore(sess, load_checkpoint_filename)
      print 'model restored'
      # TODO restore epochs counter!

    while keep_training:
      try:
        sess.run(optimizer, feed_dict={X: stft_prepared })
        if epochs % display_step == 0:
          training_cost, summary = sess.run([cost, summary_op], feed_dict={X: stft_prepared})
          print 'Training step: % 7d / % 7d' % (epochs, training_epochs), 'cost={:.9f}'.format(training_cost)
          writer.add_summary(summary, epochs)

          if training_cost <= target_cost:
            print 'target cost achieved, no more training'
            break
        epochs += 1
      except KeyboardInterrupt as e:
        print
        print
        print 'entering interactive shell'
        print 'REMEMBER not to raise SYSTEMEXIT if you want your model saved!!!'
        print 'to save and quit, assign keep_training=False'
        print
        print
        code.interact(local=locals())

    # save checkpoint
    saver.save(sess, 'checkpoints/model.epoch-%d.ckpt' % epochs)
    print 'model saved'

except:
  raise


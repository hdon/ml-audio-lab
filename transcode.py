import numpy as np
import tensorflow as tf
import scipy.signal
import scipy.io.wavfile
import code, sys, os

INFINITY = float('inf')

input_filename = sys.argv[1]
load_checkpoint_filename = sys.argv[2]
max_samples = int(sys.argv[3])

print('reading', input_filename)

def process_input(input_filename, max_samples = INFINITY):
  sample_rate, input_samples = scipy.io.wavfile.read(input_filename, mmap=True)
  if len(input_samples) > max_samples:
    input_samples = input_samples[0:max_samples]
  print(input_samples.shape)
  # collect some diagnostic info about samples
  print('input samples mean:', np.mean(input_samples))
  print('input samples std:', np.std(input_samples))
  # for our purposes we can ignore dft_freqs and dft_times
  print('stft')
  stft_freqs, stft_times, stft = scipy.signal.stft(input_samples)
  return sample_rate, stft_freqs, stft_times, stft

def foo(stft, output_filename):
  'inverse an stft and write to wav file'
  # we can ignore istft_times
  print('istft')
  istft_times, output_samples = scipy.signal.istft(stft)
  #print('converting to int')
  #output_samples = output_samples.astype(int)
  # collect some diagnostic info about samples
  print('output samples mean:', np.mean(output_samples))
  print('output samples std:', np.std(output_samples))
  #scipy.io.wavfile.write('foo.wav', 8000, output_samples)
  print('quieting')
  d = np.max(output_samples) - np.min(output_samples)
  m = np.max(output_samples) - d/2
  output_samples = (output_samples - m) / d
  print('output samples mean:', np.mean(output_samples))
  print('output samples std:', np.std(output_samples))
  print('writing')
  scipy.io.wavfile.write(output_filename, 8000, output_samples)
  #output_samples.tofile('foo.pcm')

input_tracks = []
sample_rate, stft_freqs, stft_times, stft = process_input(input_filename, max_samples)
print(stft_freqs.shape)
print(stft_times.shape)
print('stft.shape=', stft.shape)

# Hyperparameters
learning_rate = 0.1
training_epochs = 200000
target_cost = 0.000001
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
#y_ = X # autoencoder just fits output to input
#cost = tf.reduce_mean(tf.pow(y_ - y, 2))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# transposing this here seems kinda silly? maybe i should do it earlier?
# or maybe there is a more tensorflowy way to do this?
# remove imaginary components, transpose, and scale to -1,1 range
stft_prepared = np.transpose(np.real(stft))
low = np.min(stft_prepared)
high = np.max(stft_prepared)
stft_prepared = (stft_prepared - (low+high)/2.0) / ((high-low)/2.0)

# transcode!
with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1)) as sess:
  init = tf.global_variables_initializer()
  sess.run(init)
  # Load checkpoint
  if load_checkpoint_filename:
    saver = tf.train.Saver()
    saver.restore(sess, load_checkpoint_filename)
    print('model restored')
  else:
    print('need model')
    raise SystemExit(1)
  transcoded = sess.run(y, feed_dict={X: stft_prepared })

print('istft')
istft_times, output_samples = scipy.signal.istft(np.transpose(transcoded))
#print('converting to int')
#output_samples = output_samples.astype(int)
# collect some diagnostic info about samples
print('output samples mean:', np.mean(output_samples))
print('output samples std:', np.std(output_samples))
#scipy.io.wavfile.write('foo.wav', 8000, output_samples)
print('quieting')
d = np.max(output_samples) - np.min(output_samples)
m = np.max(output_samples) - d/2
output_samples = (output_samples - m) / d
print('output samples mean:', np.mean(output_samples))
print('output samples std:', np.std(output_samples))
output_filename = 'transcoded.wav'
print('writing', output_filename)
scipy.io.wavfile.write(output_filename, sample_rate, output_samples)

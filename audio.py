from math import inf
import scipy.signal
import scipy.io.wavfile
import numpy as np

# TODO
# we are using min/max to normalize our audio data, but we should probably
# be using mean + stddev. so far this is working out just fine, so, whatev

def wavToStft(input_filename, max_samples = inf):
  sample_rate, input_samples = scipy.io.wavfile.read(input_filename, mmap=True)
  #print('number of input samples:', len(input_samples))
  if len(input_samples) > max_samples:
    print('paring number of input samples down to', max_samples)
    input_samples = input_samples[0:max_samples]
  #print(input_samples.shape)
  # collect some diagnostic info about samples
  #print('input samples mean:', np.mean(input_samples))
  #print('input samples std:', np.std(input_samples))
  # for our purposes we can ignore dft_freqs and dft_times
  #print('stft')
  stft_freqs, stft_times, stft = scipy.signal.stft(input_samples)
  #print('stft shape:', stft.shape)
  #return sample_rate, stft_freqs, stft_times, stft
  stft_prepared = np.transpose(np.real(stft))
  low = np.min(stft_prepared)
  high = np.max(stft_prepared)
  stft_prepared = (stft_prepared - (low+high)/2.0) / ((high-low)/2.0)
  return sample_rate, stft_prepared

def stftToWav(stft, sample_rate, output_filename):
  stft = np.transpose(stft)
  # we can ignore istft_times
  print('istft')
  istft_times, output_samples = scipy.signal.istft(stft)
  #print('converting to int')
  #output_samples = output_samples.astype(int)
  # collect some diagnostic info about samples
  print('output samples mean:', np.mean(output_samples))
  print('output samples std:', np.std(output_samples))
  #scipy.io.wavfile.write('foo.wav', sample_rate, output_samples)
  print('quieting')
  d = np.max(output_samples) - np.min(output_samples)
  m = np.max(output_samples) - d/2
  output_samples = (output_samples - m) / d
  print('output samples mean:', np.mean(output_samples))
  print('output samples std:', np.std(output_samples))
  print('writing')
  scipy.io.wavfile.write(output_filename, sample_rate, output_samples)
  #output_samples.tofile('foo.pcm')

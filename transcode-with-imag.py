import numpy as np
import tensorflow as tf
import sys
from model import Model
from audio import wavToStft, stftToWav

input_filename = 'bar.wav'
load_checkpoint_filename = sys.argv[1]
sample_rate, stft_real, stft_imag = wavToStft(
  input_filename
, 8000 * 10
, return_imag = True
)
stft_both = np.concatenate(
  (stft_real, stft_imag)
, axis = 1
)
stft_both += 1.0

num_features = stft_both.shape[1]

model = Model(
  num_features = num_features
, num_labels = num_features
, num_hidden = [64]
, activation = [tf.tanh, tf.tanh]
, lr = 0.001
)
stft_both = model.run(load_checkpoint_filename, stft_both)
dims = int(stft_both.shape[1]/2)
stft_both -= 1.0
stft = stft_both[:,0:dims] + stft_both[:,dims:] * 1j
stftToWav(stft, sample_rate, 'transcoded.wav')

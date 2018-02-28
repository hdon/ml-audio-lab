import tensorflow as tf
import numpy as np
import math, sys
from model import Model
from trainer import Trainer
from audio import wavToStft, stftToWav

input_filename = 'bar.wav'
load_checkpoint_filename = sys.argv[1] if len(sys.argv) > 1 else None
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

tf.reset_default_graph()
model = Model(
  num_features = num_features
, num_labels = num_features
, num_hidden = [64]
, activation = [tf.tanh, tf.tanh]
, lr = 0.001
, stddev = 0.001
)
trainer = Trainer(
  model
, x=stft_both
, y=stft_both # ignored whatev
, steps_per_summary=200
, target_steps = 1600000
, target_cost = -math.inf
, load_checkpoint_filename = load_checkpoint_filename
, optimizer='adam'
, cost = 'mse'
)
trainer.train()

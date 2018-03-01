import numpy as np
import tensorflow as tf
import math
from model import Model
from trainer import Trainer
from audio import wavToStft, stftToWav

input_filename = 'bar.wav'
sample_rate, stft_real, stft_imag = wavToStft(
  input_filename
, 8000 * 10
, return_imag = True
)
stft_both = np.concatenate(
  (stft_real, stft_imag)
, axis = 1
)
num_features = stft_both.shape[1]

relu = tf.nn.relu

def reluM1(x):
  return tf.nn.relu(x) - 1.0

def addOneMorph(y):
  return y + 1

for l1 in [8, 16, 32, 64]:
  for a1 in [relu, tf.tanh, tf.sigmoid]:
    for a2 in [tf.tanh, tf.sigmoid]:
      for tries in range(4):
        print(l1, a1, a2, tries)
        tf.reset_default_graph()
        target_steps = 1600000
        model = Model(
          num_features = num_features
        , num_labels = num_features
        , num_hidden = [l1]
        , activation = [a1, a2]
        , lr_decay_steps = target_steps
        , auto_encoder = True
        )
        trainer = Trainer(
          model
        , x=stft_both
        , y=stft_both
        , steps_per_summary=200
        , target_steps = target_steps
        , target_cost = -math.inf
        , optimizer = 'adam'
        , cost = 'mse'
        )
        trainer.train()

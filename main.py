import tensorflow as tf
from model import Model
from trainer import Trainer
from audio import wavToStft, stftToWav

input_filename = 'bar.wav'
sample_rate, stft = wavToStft(input_filename, 8000 * 10)

num_features = stft.shape[1]

def reluM1(x):
  return tf.nn.relu(x) - 1.0

for l1 in [8]:
  for l2 in [8]:
    for a1 in [tf.tanh, reluM1]:
      for a2 in [tf.tanh, reluM1]:
        for a3 in [tf.tanh, reluM1]:
          for tries in range(4):
            print(l1, l2, a1, a2, a3, tries)
            tf.reset_default_graph()
            target_steps = 120
            model = Model(
              num_features = num_features
            , num_labels = num_features
            , num_hidden = [l1, l2]
            , activation = [a1, a2, a3]
            , lr_decay_steps = target_steps
            )
            trainer = Trainer(
              model
            , stft
            , None
            , steps_per_summary=40
            , target_steps = target_steps
            )
            trainer.train()

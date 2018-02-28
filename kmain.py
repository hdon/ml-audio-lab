import tensorflow as tf
import keras
from audio import wavToStft, stftToWav

input_filename = 'bar.wav'
sample_rate, stft = wavToStft(input_filename, 8000 * 10)

num_features = stft.shape[1]

relu = tf.nn.relu

def reluM1(x):
  return tf.nn.relu(x) - 1.0

def addOneMorph(y):
  return y + 1

activations = ['tanh', 'relu', 'sigmoid']

for l1 in [8]:
  for l2 in [8]:
    for a1 in activations:
      for a2 in activations:
        for a3 in activations:
          for tries in range(4):
            print(l1, l2, a1, a2, tries)
            target_steps = 8000
            model = keras.models.Sequential()
            model.add(keras.layers.core.Dense(l1, batch_input_shape=stft.shape))
            model.add(keras.layers.core.Activation(a1))
            model.add(keras.layers.core.Dense(l2))
            model.add(keras.layers.core.Activation(a2))
            model.add(keras.layers.core.Dense(stft.shape[1]))
            model.add(keras.layers.core.Activation(a3))
            model_name = '-'.join(map(str, ['k', l1, l2, a1, a2, a3, tries]))
            tensorboard = keras.callbacks.TensorBoard(
              log_dir='./log/'+model_name
            #, histogram_freq=10
            , write_grads=True
            , write_images=True
            )
            model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
            model.fit(
              stft
            , stft
            , epochs=target_steps
            , batch_size=stft.shape[0]
            , callbacks=[tensorboard]
            )

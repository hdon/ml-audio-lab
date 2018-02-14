import numpy as np
from audio import wavToStft, stftToWav
input_filename = 'bar.wav'
sample_rate, stft = wavToStft(input_filename, 8000 * 10)
print('max=', np.max(stft))
print('min=', np.min(stft))

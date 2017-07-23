import numpy as np
import scipy.signal
import scipy.io.wavfile
import code

print 'reading'
sample_rate, input_samples = scipy.io.wavfile.read('file.wav', mmap=True)
# collect some diagnostic info about samples
print 'input samples mean:', np.mean(input_samples)
print 'input samples std:', np.std(input_samples)
# for our purposes we can ignore dft_freqs and dft_times
print 'stft'
stft_freqs, stft_times, stft = scipy.signal.stft(input_samples)
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
output_samples.tofile('foo.pcm')

print 'bye'

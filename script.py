import numpy as np
from keras.models import load_model
import librosa
import librosa.display as dsp
import pylab
import matplotlib.pyplot as plt

model = load_model('model.h5')
#sound_dir = input()
# print(sound_dir)
sound_file, sample_rate = librosa.load('./a0001.wav')
mfccs = np.mean(librosa.feature.mfcc(y=sound_file, sr=sample_rate, n_mfcc=40).T, axis=0)
sound_feature = np.array(mfccs).reshape([-1,1])
sound_feature = sound_feature.reshape((1,40,1))

prediction = model.predict(sound_feature)
prediction = prediction[0][0]
ans = ''

dsp.waveplot(sound_file)
pylab.savefig('waveplot.jpg')

if prediction >= 0.50:
    ans = 'normal'
else:
    ans = 'abnormal'

print(ans)

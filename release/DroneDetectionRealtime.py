import librosa # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
import pickle
from scipy.signal import butter, lfilter # type: ignore
import noisereduce as nr # type: ignore

# -----------------------------------------------------------------
# Sample rate and desired cutoff frequencies (in Hz).
order = 4
lowcut = 200.0
highcut = 1200.0

# Filter Function 
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# -----------------------------------------------------------------
# Encoding targets
labels = ['Drone','No_Drone']           # define lables
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

print("#: Encoding targets")
print("labels:")
print(labels)

# -----------------------------------------------------------------
# Load ML classification model as a pickle file
print("Load ML classification model as a pickle file")
model_pkl_file = "/Users/panya/Project-ALL/DroneClassification/AcousticDroneDetection/model/2024-09-18 02:58:25.788779.pkl"
with open(model_pkl_file, 'rb') as file:  
    model = pickle.load(file)

# -----------------------------------------------------------------
# Audio Test Model
audio_path = "/Users/panya/Project-ALL/DroneClassification/AcousticDroneDetection/dataset/Drone/5m-100m-ex1.wav"
#print("File Audio Test: " + audio_path)
audio_test, sample_rate = librosa.load(audio_path, duration=3)  # Load audio and limit to 3 seconds

# normalize audio  
max_value = np.max(np.abs(audio_test))       # Determine the maximum values
audio_normalize = audio_test/max_value        # Use max_value and normalize sound data to get values between -1 & +1

# band pass filter 
audio_BPF = butter_bandpass_filter(audio_normalize,lowcut,highcut,sample_rate,order=7)

# Noise reduce
Audio_Reduced_Noise = nr.reduce_noise(y=audio_BPF, sr=sample_rate,prop_decrease = 1)

# convert to spectrogram 
spectrogram = librosa.feature.melspectrogram(y=Audio_Reduced_Noise, sr=sample_rate)
spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

# test model
Y_Test =  spectrogram.T.reshape(-1,1)
OutputPredic = model.predict(Y_Test.T)

print()
print('Output Predic:')
print(str(label_encoder.inverse_transform(OutputPredic)))
print()

"""
# Plot Spectrogram
plt.figure(figsize=(12, 4))
plt.suptitle(f'Example for Spectrogram')
plt.subplot(1, 2, 1)
plt.title(f'Spectrogram')
librosa.display.specshow(spectrogram, x_axis='time', y_axis='hz',cmap='viridis')  #cmap = 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
plt.colorbar(format='%+2.0f dB')
plt.xlabel('Time')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.title(f'Audio Waveform')
plt.plot(np.linspace(0, len(Audio_Reduced_Noise) / sample_rate, len(Audio_Reduced_Noise)), Audio_Reduced_Noise)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
"""
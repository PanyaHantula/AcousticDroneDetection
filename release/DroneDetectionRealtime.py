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


    
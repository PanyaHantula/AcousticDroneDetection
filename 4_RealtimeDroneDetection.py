import librosa 
import librosa.display 
import numpy as np
import noisereduce as nr
from tensorflow.keras.models import load_model # type: ignore
import pickle
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
import sounddevice as sd # type: ignore

#--------------------------------------------------------------------------
def spectrogram_cal(data,fs):
    ms = librosa.feature.melspectrogram(y=data, sr=fs, n_fft=2048, hop_length=128, n_mels=256)
    spectrogram_db = librosa.power_to_db(ms, ref=np.max)
    
    return spectrogram_db

#--------------------------------------------------------------------------
# Load CNN Model
print('-----------------------------------------')
print("#: load CNN model")
myModel = load_model('model\\model_01\\myModel.h5') 
myModel.summary()

# load config
print('-----------------------------------------')
print("#: load label config")

with open ('model\\model_01\\labels', 'rb') as fp:
    labels = pickle.load(fp)

print("labels : " + str(labels))
# Encode target labels
label_encoder = LabelEncoder()
label_encoder.fit_transform(labels)

print('-----------------------------------------')
print("#: Running the real-time audio classification.....")
# Real-time audio classification parameters
sample_rate = 22050  # Sampling rate
duration = 1  # Duration of each audio frame in seconds
frame_length = int(sample_rate * duration)

def extract_mel_spectrogram(audio, sr):
     #### Converts audio signal to a mel spectrogram. ####
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=128, n_mels=256)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db

def predict_audio_class(audio_frame):
    #### Predicts the class of an audio frame. ####
    mel_spectrogram = extract_mel_spectrogram(audio_frame, sample_rate) / 255.0
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)  # Add batch dimension
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)  # Add channel dimension

    prediction = myModel.predict(mel_spectrogram)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class

def audio_callback(indata, frames, time, status):
    #### Callback to process incoming audio. ####
    if status:
        print(f"Audio Status: {status}")
    audio_frame = indata[:, 0]  # Get mono channel
    if len(audio_frame) == frame_length:
        class_prediction = predict_audio_class(audio_frame)
        lable_Output = label_encoder.inverse_transform(class_prediction)
        print(f"Predicted Class: {class_prediction}")
        print(f'Predicted Lable: {lable_Output}')

if __name__ == '__main__':
    print("System Start...")
    with sd.Stream(channels=1, callback=audio_callback):
        sd.sleep(60000)
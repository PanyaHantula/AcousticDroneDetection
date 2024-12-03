import sounddevice as sd
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
import pickle
from sklearn.preprocessing import LabelEncoder # type: ignore
import librosa 

#--------------------------------------------------------------------------
# Load CNN Model
print('-----------------------------------------')
print("#: load CNN model")
myModel = load_model('D:\\SF-67\\model\\20241130164341\\myModel.h5') 
myModel.summary()

# load config
print('-----------------------------------------')
print("#: load label config")

with open ('D:\\SF-67\\model\\20241130164341\\labels', 'rb') as fp:
    labels = pickle.load(fp)

print("labels : " + str(labels))
# Encode target labels
label_encoder = LabelEncoder()
label_encoder.fit_transform(labels)

#--------------------------------------------------------------------------
def extract_mel_spectrogram(audio, sr=22050):
     #### Converts audio signal to a mel spectrogram. ####
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=128, n_mels=256)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    print(f'Shape of spectrogram : {mel_spectrogram_db.shape }')
    return mel_spectrogram_db

def predict_audio_class(audio_frame):
    #### Predicts the class of an audio frame. ####
    mel_spectrogram = extract_mel_spectrogram(audio_frame) / 255.0
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)  # Add batch dimension
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)  # Add channel dimension

    prediction = myModel.predict(mel_spectrogram)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class

def run():
    print("Listening... ")
    
    samplerate = 22050  # Whisper expects 16kHz audio
    chunk_duration = 1  # Duration of each audio chunk in seconds
    frame_length = int(samplerate * chunk_duration)

    # A list to store the recorded audio chunks
    audio_buffer = []
    try:
        while True:
            # Record a chunk of audio
            print(f"Recording {chunk_duration} seconds...")
            chunk = sd.rec(int(samplerate * chunk_duration), samplerate=samplerate, channels=1, dtype='float32')
            sd.wait()  # Wait for the chunk to finish recording

            # predict the concatenated audio
            print("predicting audio...")
            audio_buffer = np.squeeze(chunk)
            class_prediction = predict_audio_class(audio_buffer)
            lable_Output = label_encoder.inverse_transform(class_prediction)
            print(f"Predicted Class: {class_prediction}")
            print(f'Predicted Lable: {lable_Output}')
            print()
        
    except Exception as e:
        print("An error occurred:", e)

if __name__ == '__main__':
    print("System Start...")
    while True:
        run()

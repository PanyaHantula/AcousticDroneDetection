import sounddevice as sd
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
import pickle
from sklearn.preprocessing import LabelEncoder 
import librosa 
import librosa.display 
import matplotlib.pyplot as plt 

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

from scipy.fft import fft, fftfreq # type: ignore

# ----- 1-D discrete Fourier transforms ------
def audioFFT_cal (data):
    N = int (chunk_duration * samplerate)        #   Number of sample points

    T = 1.0 / (samplerate)   # sample spacing
    x = np.linspace(0.0, N*T, N, endpoint=False)
    yf = fft(data)
    Xf = fftfreq(N, T)[:N//2]
    FFT_Amplitude = 10*np.log(np.abs(yf[0:N//2]))
    
    return Xf,FFT_Amplitude

#--------------------------------------------------------------------------
def extract_mel_spectrogram(audio):
     #### Converts audio signal to a mel spectrogram. ####
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=samplerate, n_fft=2048, hop_length=128, n_mels=256)
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


def PlotAudioGraph (y_signal,title):
    plt.figure(figsize=(10, 8))
    plt.suptitle(f'Predicte Output : {str(title[0])}',fontweight="bold", size=20)

    # ----- Plot Audio Waveform  -----
    plt.subplot(2, 2, 1)
    plt.title(f'Audio Waveform')
    plt.plot(np.linspace(0, len(y_signal) / samplerate, len(y_signal)), y_signal)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid()
    # ----- Plot FFT  -----
    xf,yf = audioFFT_cal(y_signal)    
    plt.subplot(2, 2, 2)
    plt.title(f'FFT waveform')
    plt.plot(xf, yf)
    plt.grid()
    plt.xlabel('Freq (Hz)')
    plt.ylabel('Normalize Amplitude (dB)')
    plt.ylim(-70,80)

    # ------- Plot Spectrogram ---------
    spectrogram_db = extract_mel_spectrogram(y_signal)
    plt.subplot(2, 1, 2)
    plt.title(f'Spectrogram')
    librosa.display.specshow(spectrogram_db, sr=samplerate, x_axis='time', y_axis='linear', cmap='viridis')
    #cmap = 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram shape {spectrogram_db.shape}')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.grid()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def run():
    print("Listening... ")
    
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
            PlotAudioGraph(audio_buffer,lable_Output)
            print()
        
    except Exception as e:
        print("An error occurred:", e)

if __name__ == '__main__':
    global samplerate
    global chunk_duration

    samplerate = 22050  # Whisper expects 16kHz audio
    chunk_duration = 1  # Duration of each audio chunk in seconds

    print("System Start...")
    while True:
        run()

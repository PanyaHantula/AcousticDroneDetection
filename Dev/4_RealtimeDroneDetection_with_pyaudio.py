import sounddevice as sd
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
import pickle
from sklearn.preprocessing import LabelEncoder 
import librosa 
import librosa.display 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation

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
samplerate = 22050  # Whisper expects 16kHz audio
chunk_duration = 1  # Duration of each audio chunk in seconds

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
   # print(f'Shape of spectrogram : {mel_spectrogram_db.shape }')
    return mel_spectrogram_db

def predict_audio_class(audio_frame):
    #### Predicts the class of an audio frame. ####
    mel_spectrogram = extract_mel_spectrogram(audio_frame) / 255.0
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)  # Add batch dimension
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)  # Add channel dimension

    PredictionProb = myModel.predict(mel_spectrogram)
    predicted_class = np.argmax(PredictionProb, axis=1)
    return PredictionProb,predicted_class


def initGraph(y_signal):
    global TimeDomainGraph, FrqeDomainGraph ,SpectrogramGraph, Maintitle, figure

    plt.ion()  # turning interactive mode on
    
    figure = plt.figure(figsize=(10, 8))
    figure.suptitle(f'Prediction Output : -',fontweight="bold", size=20)

    gs = figure.add_gridspec(2,2)
    ax1 = figure.add_subplot(gs[0, 0])
    ax2 = figure.add_subplot(gs[0, 1])
    ax3 = figure.add_subplot(gs[1, :])

    #figure, ax = plt.subplots(2,2,figsize=(10, 8),squeeze=False)
    
    # ----- Plot Audio Waveform  -----
    audioTimespace = np.linspace(0, len(y_signal) / samplerate, len(y_signal))
    ax1.title.set_text('Audio Waveform')
    ax1.set_ylabel('Normalize Amplitude')
    ax1.set_xlabel('Time (s)')
    ax1.grid()
    TimeDomainGraph, = ax1.plot(audioTimespace, y_signal)

    # ----- Plot FFT  -----
    xf,yf = audioFFT_cal(y_signal)   
    ax2.title.set_text('FFT waveform')
    ax2.set_ylabel('Normalize Amplitude (dB)')
    ax2.set_xlabel('Freq (Hz)')
    ax2.grid()
    ax2.set_ylim(-70,80)
    FrqeDomainGraph, = ax2.plot(xf, yf)

    # ------- Plot Spectrogram ---------
    spectrogram_db = extract_mel_spectrogram(y_signal)
    ax3.title.set_text(f'Spectrogram shape {spectrogram_db.shape}')
    ax3.set_ylabel('Frequency (Hz)')
    ax3.set_xlabel('Time (s)')
    ax3.grid()
    SpectrogramGraph = ax3.imshow(spectrogram_db,interpolation='nearest', aspect='auto')
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])

def UpdatGraph(y_signal,title):
    # plotting newer graph
    TimeDomainGraph.set_ydata(y_signal)

    xf,yf = audioFFT_cal(y_signal) 
    FrqeDomainGraph.set_xdata(xf)
    FrqeDomainGraph.set_ydata(yf)

    spectrogram_db = extract_mel_spectrogram(y_signal)
    SpectrogramGraph.set_data(spectrogram_db)

    figure.suptitle(f'Prediction Output : {str(title[0])}',fontweight="bold", size=20)

    figure.canvas.draw()
    figure.canvas.flush_events()
    # plt.pause(0.01)

import pyaudio
import numpy as np

FRAMES_PER_BUFFER = 1000
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 22050
p = pyaudio.PyAudio()

def record_audio():
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER
    )

    #print("start recording...")

    frames = []
    seconds = 1
    for i in range(0, int(RATE / FRAMES_PER_BUFFER * seconds)):
        data = stream.read(FRAMES_PER_BUFFER)
        frames.append(data)

    # print("recording stopped")

    stream.stop_stream()
    stream.close()
    
    return np.frombuffer(b''.join(frames), dtype=np.float32)

def run():
    print("Listening... ")
    audio_buffer = []

    try:    
        audio_buffer = record_audio()
        initGraph(audio_buffer)

        while True:
            audio_buffer = record_audio()
            # predict the concatenated audio
            print("predicting audio...")
            #audio_buffer = np.squeeze(audio_buffer)
            PredictionProb,class_prediction = predict_audio_class(audio_buffer)
            lable_Output = label_encoder.inverse_transform(class_prediction)
            print(f"Predicted Prob: {PredictionProb}")
            print(f"Predicted Class: {class_prediction}")
            print(f'Predicted Lable: {lable_Output}')
            print()

            UpdatGraph(audio_buffer,lable_Output)

    except Exception as e:
        print("An error occurred:", e)

if __name__ == '__main__':
    print("System Start...")
    run()

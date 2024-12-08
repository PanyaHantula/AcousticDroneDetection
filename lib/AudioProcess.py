import numpy as np
import librosa 
from scipy.fft import fft, fftfreq
import sounddevice as sd
import noisereduce as nr       

#-----------------------------------------------------------------------
stationary=True
prop_decrease=1
n_std_thresh_stationary = 1

samplerate = 22050  
TimeRecord = 1

# ----- 1-D discrete Fourier transforms ------
def audioFFT_cal (data):
    N = int (TimeRecord * samplerate)        #   Number of sample points

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

def LoadAudioFile(AudioFilePart):
    print("#: Load the Drone audio file")
    audio_file = AudioFilePart  # Replace with your audio file path
    audio_signal, fs = librosa.load(audio_file) 
    audio_signal = audio_signal[:22000]
    timesDuration = librosa.get_duration(y=audio_signal, sr=fs)
    
    print('- normalize audio')  
    max_value = np.max(np.abs(audio_signal))       # Determine the maximum values
    audio_signal = audio_signal/max_value           # Use max_value and normalize sound data to get values between -1 & +1

    print(f'- Sampling Rate: {fs} Hz')
    print(f'- Audio Duration: {timesDuration:.0f} seconds')
    print('#: Done !!')
    print()

    TimeSpace = np.linspace(0, timesDuration, len(audio_signal))

    return audio_signal,TimeSpace

def AudioCapture():
    print("Listening... ")
    audio_buffer = []

    try:   
        # init first graph   
        audio_buffer = sd.rec(int(samplerate * TimeRecord), samplerate=samplerate, channels=1, dtype='float32')
        sd.wait()  
        audio_buffer = np.squeeze(audio_buffer)

        # normalize audio  
        max_value = np.max(np.abs(audio_buffer))       # Determine the maximum values
        audio_normalize = audio_buffer/max_value        # Use max_value and normalize sound data to get values between -1 & +1

        # perform noise reduction
        audio_reduced_noise = nr.reduce_noise(y=audio_normalize, 
                                            sr=samplerate, 
                                            stationary=stationary, 
                                            prop_decrease=prop_decrease,
                                            n_std_thresh_stationary=n_std_thresh_stationary)    # ,use_torch=True )

        TimeSpace = np.linspace(0, TimeRecord, len(audio_reduced_noise))

        return audio_reduced_noise,TimeSpace
    
    except Exception as e:
        print("An error occurred:", e)
        

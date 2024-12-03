import sounddevice as sd # type: ignore

# Real-time audio classification parameters
sample_rate = 22050  # Sampling rate
duration = 1  # Duration of each audio frame in seconds
frame_length = int(sample_rate * duration)

def callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    outdata[:] = indata
    print(len(outdata))
    
if __name__ == '__main__':
    print("System Start...")
    try:
        with sd.RawStream(channels=2, dtype='int32', callback=callback):
            print('#' * 80)
            print('press Return to quit')
            print('#' * 80)
            input()
    except KeyboardInterrupt:
        print('\nInterrupted by user')

#     stream = sd.InputStream(channels=1, dtype='float32', callback=capture_audio,
#                            samplerate=SAMPLING_RATE, blocksize=BLOCK_SIZE)
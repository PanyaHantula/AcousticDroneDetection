import sys
from PySide6.QtGui import QGuiApplication, QFont
from PySide6.QtCore import Qt, QSize, QTimer, QTime
from PySide6.QtWidgets import QApplication, QMainWindow, \
    QGridLayout, QHBoxLayout, QVBoxLayout, \
    QWidget, QPushButton, QLineEdit, QCheckBox, QLabel, QGroupBox

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np
from random import randint
import pyaudio 
import struct 
from scipy.fftpack import fft

class MainPage (QWidget):
    def __init__(self):
        super().__init__()

        # pyaudio stuff
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 1024 * 2

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            output=True,
            frames_per_buffer=self.CHUNK,
        )
        # waveform and spectrum x points
        self.x = np.arange(0, 2 * self.CHUNK, 2)
        self.f = np.linspace(0, self.RATE // 2, self.CHUNK // 2)
        
    def InitAudioGraph(self):
        self.setWindowTitle('Real Time Audio Visualize')
        self.setFixedSize(QSize(1024,600))
    
        #Create hBox1
        GraphBox = QHBoxLayout()
        self.setLayout(GraphBox)

        # ------------------------------------------------------------ #
        # Creat hBox Graph No.1 (Left Layer)
        AudioWaveformGraphVBoxLeft = QVBoxLayout()
        GraphBox.addLayout(AudioWaveformGraphVBoxLeft)

        self.GraphWaveformAudio = pg.PlotWidget()           # Creat Graph Widget
        AudioWaveformGraphVBoxLeft.addWidget(self.GraphWaveformAudio)       # add vBox for  in hBox1
        wf_data = np.array([randint(0,256) for _ in range(len(self.x))])   # Dummy data to plot graph

        # plot data: x, y values
        self.GraphWaveformAudio.setBackground('#FFFFFF')        # Background Colour
        pen = pg.mkPen(color=(255, 0, 0),width=5)               # Line Colour, Width & Style
        self.GraphWaveformAudio.setTitle("Audio Waveform", color="b", size="20pt")    # Plot Titles
        
        styles = {'color':'r', 'font-size':'20px'}      # Axis Labels
        self.GraphWaveformAudio.setLabel('left', 'Amplitude', **styles)
        self.GraphWaveformAudio.setLabel('bottom', 'Time (Sec)', **styles) 
        self.GraphWaveformAudio.showGrid(x=True, y=True)# Background Grid
        self.GraphWaveformAudio.setYRange(0, 255, padding=0)
        self.GraphWaveformAudio.setXRange(0, 2 * self.CHUNK, padding=0.005)
        self.data_line_waveform =  self.GraphWaveformAudio.plot(self.x, wf_data, pen=pen)

        # ------------------------------------------------------------ #
        # Creat HBox Graph No.2 (Right Layer)
        AudioWaveformGraphVBoxRight = QVBoxLayout()
        GraphBox.addLayout(AudioWaveformGraphVBoxRight)

        self.GraphFFTAudio = pg.PlotWidget()           # Creat Graph Widget
        AudioWaveformGraphVBoxLeft.addWidget(self.GraphFFTAudio)       # add vBox for  in hBox1

        # Dummy data to plot graph
        self.FFT = np.array([randint(0,1024) for _ in range(len(self.f))])   # Dummy data to plot graph
        sp_data = fft(np.array(self.FFT, dtype='int8') - 128)
        sp_data = np.abs(sp_data[0:int(self.CHUNK / 2)]
                         ) * 2 / (128 * self.CHUNK)
        
        # plot data: x, y values
        self.GraphFFTAudio.setBackground('#FFFFFF')        # Background Colour
        pen = pg.mkPen(color=(0, 0, 255),width=5)               # Line Colour, Width & Style
        self.GraphFFTAudio.setTitle("Audio Spectrum", color="b", size="20pt")    # Plot Titles
        
        styles = {'color':'r', 'font-size':'20px'}      # Axis Labels
        self.GraphFFTAudio.setLabel('left', 'Amplitude', **styles)
        self.GraphFFTAudio.setLabel('bottom', 'Time (Sec)', **styles) 
        self.GraphFFTAudio.showGrid(x=True, y=True)# Background Grid
        self.GraphFFTAudio.setLogMode(x=True, y=True)
        self.GraphFFTAudio.setYRange(-4, 0, padding=0)
        self.GraphFFTAudio.setXRange(np.log10(20), np.log10(self.RATE / 2), padding=0.005)
        self.data_line_FFT =  self.GraphFFTAudio.plot(self.f, sp_data, pen=pen)

    def AudioStream(self):
        # Capture Audio data that are update the graph of waveform
        #wf_data = self.stream.read(self.CHUNK,)
        #self.stream.stop_stream()
        #self.stream.close()
        #wf_data = struct.unpack(str(2 * self.CHUNK) + 'B', wf_data)
        #wf_data = np.array(wf_data, dtype='b')[::2] + 128
        wf_data = np.array([randint(0,256) for _ in range(len(self.x))])   # Dummy data to plot graph
        
        # plot waveform data
        self.data_line_waveform.setData(self.x, wf_data)  # Update waveform data.

        # plot FFT data 
        sp_data = fft(np.array(wf_data, dtype='int8') - 128)
        sp_data = np.abs(sp_data[0:int(self.CHUNK / 2)]
                        ) * 2 / (128 * self.CHUNK)
        self.data_line_FFT.setData(self.f, sp_data)  # Update FFT data.

if __name__ == '__main__':
    app = QApplication([])
    DroneDetectionWindown = MainPage()
    DroneDetectionWindown.InitAudioGraph()

    timer = QTimer()
    timer.setInterval(20)
    timer.timeout.connect(DroneDetectionWindown.AudioStream)
    timer.start()
    
    #DroneDetectionWindown.AudioStream()
    DroneDetectionWindown.show()
    app.exec()

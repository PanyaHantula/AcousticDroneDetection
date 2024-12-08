import os
import sys
import numpy as np
from PySide6.QtCore import QTimer,Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtGui import QColor, QPalette

from lib.AudioProcess import LoadAudioFile,AudioCapture,audioFFT_cal,extract_mel_spectrogram

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# ============================================== #
#                Class MainWindows               #
# ============================================== #
class MainWindows(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Realtime Drone Detection System")
        self.setGeometry(100, 100, 1024, 800)  # x, y, width, height
        self.initUI()

    def initUI(self):
        # --- Graph Layout ----
        MainLayout = QHBoxLayout()
        GraphLayout = QVBoxLayout()
        SideManuLayout = QVBoxLayout() 

        # --------------------- Graph Layout -----------------------------------
        MainLayout.addLayout(GraphLayout)       # Add Layout
        MainLayout.setContentsMargins(0,0,0,0)
        MainLayout.setSpacing(20)

        # add WaveformGraph Widget 
        WaveformGraph = FigureCanvas(Figure(figsize=(6, 2)))
        GraphLayout.addWidget(WaveformGraph)
        audio_signal,TimeSpace = AudioCapture()
        self._Waveform_ax = WaveformGraph.figure.subplots()
        self._Waveform_ax.set_title("Waveform")
        self._Waveform_ax.set_ylabel("Normalize Amplitude")
        self._Waveform_ax.set_ylim(-1.5,1.5)
        self._Waveform_ax.grid()
        self._WaveformPlot, = self._Waveform_ax.plot(TimeSpace, audio_signal,'g')

        # add FFTGraph Widget 
        FFTGraph = FigureCanvas(Figure(figsize=(6, 2)))
        GraphLayout.addWidget(FFTGraph)
        xf,yf = audioFFT_cal(audio_signal) 
        self._FFTPlot_ax = FFTGraph.figure.subplots()
        self._FFTPlot_ax.set_title("FFT")
        self._FFTPlot_ax.set_ylabel("Amplitude (dB)")
        self._FFTPlot_ax.set_ylim(-120,60)
        self._FFTPlot_ax.grid()
        self._FFTPlot, = self._FFTPlot_ax.plot(xf,yf)


        SpectrogramGraph = FigureCanvas(Figure(figsize=(6, 4)))
        GraphLayout.addWidget(SpectrogramGraph)
        spectrogram_db = extract_mel_spectrogram(audio_signal)
        print(spectrogram_db.shape)
        self._Spectrogramlot_ax = SpectrogramGraph.figure.subplots()
        self._SpectrogramPlot =  self._Spectrogramlot_ax.imshow(spectrogram_db,interpolation='nearest', aspect='auto')

        # --------------------- Side Manu Layout -----------------------------------
        MainLayout.addLayout(SideManuLayout)    # Add Layout

        # add Start Button Widget 
        btnStart = QPushButton("START")
        SideManuLayout.addWidget(btnStart)

        # add Start Button Widget 
        FreqLable = QLabel("Frequency: ")
        FreqLable.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        SideManuLayout.addWidget(FreqLable)

        AmplitudeLable = QLabel("Amplitude: ")
        AmplitudeLable.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        SideManuLayout.addWidget(AmplitudeLable)

        # -------------------------------------------------------------------------------
        #  launch Layout 
        widget = QWidget()
        widget.setLayout(MainLayout)
        self.setCentralWidget(widget)
        
        # # -------------------------------------------------------------------------------
        # ---- Set Timer for Update Grpah -----
        timer = QTimer(self)
        timer.timeout.connect(self._update_graph)
        timer.start(1000)

    def _update_graph(self):
        audio_signal,TimeSpace = AudioCapture()
        self._WaveformPlot.set_data(TimeSpace,audio_signal)
        self._WaveformPlot.figure.canvas.draw()

        xf,yf = audioFFT_cal(audio_signal) 
        self._FFTPlot.set_data(xf,yf)
        self._FFTPlot.figure.canvas.draw()

        spectrogram_db = extract_mel_spectrogram(audio_signal)
        self._SpectrogramPlot.set_data(spectrogram_db)
        self._SpectrogramPlot.figure.canvas.draw()

# Run the application
app = QApplication(sys.argv)
window = MainWindows()
window.show()
sys.exit(app.exec_())

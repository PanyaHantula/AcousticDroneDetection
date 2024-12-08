import sys
import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MainWindows(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Iris Dataset Visualizations")
        self.setGeometry(100, 100, 800, 600)  # x, y, width, height

        self.initUI()

    def initUI(self):
        # Set Layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Add Graph1 to H1Layout
        static_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(static_canvas)

        # Add Graph2 to H1Layout
        dynamic_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(dynamic_canvas)
        
        # set data of static_ax 
        self._static_ax = static_canvas.figure.subplots()
        t = np.linspace(0, 10, 501)
        self._static_ax.plot(t, np.tan(t), ".")

        self._dynamic_ax = dynamic_canvas.figure.subplots()
        x = np.random.randn(101)
        t = np.linspace(0, 10, 101)
        self._line, = self._dynamic_ax.plot(t,x)
        #self._dynamic_ax.plot(t,x)
        
        # Set Timer 
        timer = QTimer(self)
        timer.timeout.connect(self._update_canvas)
        timer.start(1000)

    def _update_canvas(self):
        t = np.linspace(0, 10, 101)
        self._line.set_data(t,np.random.randn(101))
        self._line.figure.canvas.draw()
            
# Run the application
app = QApplication(sys.argv)
window = MainWindows()
window.show()
sys.exit(app.exec_())

"""
class IrisVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Iris Dataset Visualizations")
        self.setGeometry(100, 100, 1200, 900)  # x, y, width, height

        self.initUI()

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.canvas)
        
    def initUI(self):
        self.canvas = FigureCanvas(Figure(figsize=(6, 4)))
        ax = self.canvas.figure.add_subplot(111)
        minutes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        temperature = [30, 32, 34, 32, 33, 31, 29, 32, 35, 30]
        ax.plot(minutes,temperature)
        # sns.scatterplot(data=iris, x="sepal_length", y="sepal_width", hue="species", ax=ax)
        # ax.set_title("Sepal Length vs Sepal Width")

"""
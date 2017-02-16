#_*_coding:utf-8_*_
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as figureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import sys


class DrawWidget(QWidget):
    def __init__(self,parent=None):
        super(DrawWidget,self).__init__(parent)
        figure = plt.gcf() #返回当前的figure
        self.canvas = figureCanvas(figure)
        x = [1,2,3]
        y = [4,5,6]
        plt.plot(x,y)
        plt.title('Example')
        plt.xlabel('x')
        plt.ylabel('y')
        self.canvas.draw()
        layout = QHBoxLayout(self)
        layout.addWidget(self.canvas)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = DrawWidget()
    ui.show()
    app.exec_()
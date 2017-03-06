# -*- coding: utf-8 -*-
import sys
from PyQt4 import QtGui, QtCore


class buttonRedrect(QtGui.QWidget):
    def __init__(self):
        super(buttonRedrect, self).__init__()
        self.setWindowTitle('Mouse Event')
        self.setWindowIcon(QtGui.QIcon('QQ.png'))
        self.labels = QtGui.QLabel(u'点我试试！', self)
        self.labels.setGeometry(50, 50, 150, 50)
        self.labels.mouseReleaseEvent = self.events

    def events(self, event):
        ev = event.button()
        if ev == QtCore.Qt.LeftButton:
            OK = QtGui.QMessageBox.information(self, (u'提示'), (u'左键'), QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
            if OK == QtGui.QMessageBox.Yes:
                QtGui.QMessageBox.information(self, (u'提示'), (u'YES'), QtGui.QMessageBox.Yes)
            else:
                QtGui.QMessageBox.information(self, (u'提示'), (u'NO'), QtGui.QMessageBox.Yes)
        elif ev == QtCore.Qt.RightButton:
            OK = QtGui.QMessageBox.warning(self, (u'提示'), (u'右键'), QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
        elif ev == QtCore.Qt.MiddleButton:
            OK = QtGui.QMessageBox.question(self, (u'提示'), (u'滚动轴'), QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)


app = QtGui.QApplication(sys.argv)
buttonr = buttonRedrect()
buttonr.show()
sys.exit(app.exec_())

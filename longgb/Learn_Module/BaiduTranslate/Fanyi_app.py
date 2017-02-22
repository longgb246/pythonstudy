# -*- coding: utf-8 -*-
import sys
from PyQt4 import QtGui, QtCore


class buttonRedrect(QtGui.QWidget):                       # 继承
    def __init__(self):
        super(buttonRedrect, self).__init__()               # 继承
        self.initUI()

    def initUI(self):
        # ===================================== 字体、提示 ===============================
        QtGui.QToolTip.setFont(QtGui.QFont('SansSerif', 10))    # 设置提示字体
        self.setToolTip('This is a <b>QWidget</b> widget')      # 提示字体格式，可以使用html标签

        # ===================================== 窗口 =====================================
        self.setWindowTitle(u'百度翻译')                        # 设置窗口名称
        self.setWindowIcon(QtGui.QIcon(r"D:\Life\Photo\favicon.ico"))       # 设置Icon的图标
        self.setGeometry(300, 300, 300, 150)
        # self.resize(300, 150)                                 # 设置窗口大小
        # self.move(300, 300)                                   # 设置位置
        # self.center()                                         # 设置居中，见下面自定义函数。




        # ===================================== 按钮 =====================================
        self.okButton = QtGui.QPushButton("OK", self)           # 设置按钮
        self.okButton.setToolTip('This is a <b>QPushButton</b> widget')     # 设置按钮提示
        # self.okButton.resize(btn.sizeHint())                  # sizeHint()方法返回一个推荐的大小。
        # qbtn = QtGui.QPushButton('Quit', self)
        # qbtn.clicked.connect(QtCore.QCoreApplication.instance().quit)     # 按钮连接关闭的方法。关闭当前窗口
        # qbtn.clicked.connect(self.btnCloseEvent)
        self.cancelButton = QtGui.QPushButton("Cancel")




        self.labels = QtGui.QLabel(u'点我试试！', self)
        # self.labels.setGeometry(50, 50, 150, 50)
        # self.labels.mouseReleaseEvent = self.events


        hbox = QtGui.QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.okButton)
        # hbox.addWidget(self.cancelButton)
        vbox = QtGui.QVBoxLayout()
        # vbox.addStretch(1)
        vbox.addLayout(hbox)
        self.setLayout(vbox)

        self.okButton.mouseReleaseEvent = self.events


    def center(self):
        '''
        使得窗口居中屏幕
        '''
        # ===================================== QDesktopWidget ===========================
        # QtGui.QDesktopWidget这个类提供了用户桌面的信息，包括屏幕大小。
        qr = self.frameGeometry()                               # 用frameGeometry方法得到了主窗口的矩形框架qr。
        cp = QtGui.QDesktopWidget().availableGeometry().center()            # 调用这些方法来得到屏幕分辨率，并最终得到屏幕中间点的坐标cp。
        qr.moveCenter(cp)                                       # 将矩形框架移至屏幕正中央，大小不变。
        self.move(qr.topLeft())                                 # 窗口移至矩形框架的左上角点，这样应用窗口就位于屏幕的中央了【注意部件的move都是左上角移动到某点】。

    def events(self, event):
        '''
        点击事件，弹出提示信息。
        '''
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

    def closeEvent(self, event):
        '''
        重写了这个关闭的方法，跳出提示的信息，不用设置链接按钮
        '''
        # ===================================== 提示信息 =================================
        reply = QtGui.QMessageBox.question(self, u'警告', u"你确定要退出翻译吗?", QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No)
        if reply == QtGui.QMessageBox.Yes:
            event.accept()              # 这个事件会被接受
        else:
            event.ignore()              # 这个事件会被接受

    def btnCloseEvent(self, event):
        # 目前出了点问题，还没有实现
        reply = QtGui.QMessageBox.question(self, u'警告', u"你确定要退出翻译吗?", QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No)
        if reply == QtGui.QMessageBox.Yes:
            pass
        else:
            pass



app = QtGui.QApplication(sys.argv)
buttonr = buttonRedrect()
buttonr.show()
sys.exit(app.exec_())

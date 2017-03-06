# -*- coding: utf-8 -*-
from PyQt4 import QtGui, QtCore
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import json


class zhengRenApp(QtGui.QWidget):
    def __init__(self, start_x=300):
        super(zhengRenApp, self).__init__()  # 继承
        self.start_x = start_x
        self.initUI()

    def initUI(self):
        # ===================================== 窗口 =====================================
        self.setWindowTitle(u'于函好美丽!')  # 设置窗口名称
        self.setWindowIcon(QtGui.QIcon(r"D:\Life\Photo\favicon.ico"))  # 设置Icon的图标
        self.setGeometry(self.start_x, self.start_x, 300, 60)
        # self.resize(300, 150)                                 # 设置窗口大小
        # self.move(300, 300)                                   # 设置位置
        # self.center()  # 设置居中，见下面自定义函数。

        # ===================================== 按钮 =====================================
        self.okButton = QtGui.QPushButton(u"关闭", self)           # 设置按钮
        self.okButton.mouseReleaseEvent = self.closeEvents         # 设置button的点击事件
        self.okButton.resize(150, 60)
        self.okButton2 = QtGui.QPushButton(u"不关闭", self)        # 设置按钮
        self.okButton2.mouseReleaseEvent = self.closeEvents      # 设置button的点击事件
        self.okButton2.resize(150, 60)
        self.okButton2.move(150, 0)

    def zhengRen(self, event):
        ev = event.button()

    def closeEvent(self, event):
        event.ignore()
        aa = zhengRenApp(self.start_x + 20)
        aa.show()

    def closeEvents(self, event):
        '''
        点击事件，弹出提示信息。
        '''
        aa = zhengRenApp(self.start_x + 20)
        aa.show()
        # ev = event.button()
        # OK = QtGui.QMessageBox.warning(self, (u'提示'), (u'左键'), (u'左键'))
        # OK.setGeometry(self.start_x, self.start_x, 300, 60)
        # if OK == 0:
        #     QtGui.QMessageBox.information(self, (u'提示'), (u'YES'), QtGui.QMessageBox.Yes)
        # if ev == QtCore.Qt.LeftButton:
        #     OK = QtGui.QMessageBox.information(self, (u'提示'), (u'左键'), QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
        #     if OK == QtGui.QMessageBox.Yes:
        #         QtGui.QMessageBox.information(self, (u'提示'), (u'YES'), QtGui.QMessageBox.Yes)
        #     else:
        #         QtGui.QMessageBox.information(self, (u'提示'), (u'NO'), QtGui.QMessageBox.Yes)
        # elif ev == QtCore.Qt.RightButton:
        #     OK = QtGui.QMessageBox.warning(self, (u'提示'), (u'右键'), QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
        # elif ev == QtCore.Qt.MiddleButton:
        #     OK = QtGui.QMessageBox.question(self, (u'提示'), (u'滚动轴'), QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)

    def nocloseEvents(self, event):
        ev = event.button()
        OK = QtGui.QMessageBox.warning(self, (u'提示'), (u'左键'), (u'左键'))

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


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    fanyi = zhengRenApp()
    fanyi.show()
    sys.exit(app.exec_())

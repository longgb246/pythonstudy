# -*- coding: utf-8 -*-
# from PyQt4 import QtGui, QtCore
from PyQt5 import QtGui, QtCore
import httplib
import md5
import urllib
import random
import chardet
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import json

# ==================================== 配置翻译 ====================================
appid = '20170222000039667'                     # 账号
secretKey = 'IeZ2wtwnNYxszXPDYDzC'              # 密钥
httpClient = None
myurls = '/api/trans/vip/translate'


class fanyiApp(QtGui.QWidget):                       # 继承
    def __init__(self):
        super(fanyiApp, self).__init__()               # 继承
        self.initUI()

    def initUI(self):
        # ===================================== 字体、提示 ===============================
        QtGui.QToolTip.setFont(QtGui.QFont('SansSerif', 10))    # 设置提示字体
        # self.setToolTip('This is a <b>QWidget</b> widget')      # 提示字体格式，可以使用html标签

        # ===================================== 窗口 =====================================
        self.setWindowTitle(u'百度翻译——函函版本')                        # 设置窗口名称
        self.setWindowIcon(QtGui.QIcon(r"./favicon.ico"))                  # 公司家里通用
        # self.setWindowIcon(QtGui.QIcon(r"D:\Life\Photo\favicon.ico"))       # 设置Icon的图标【家里】
        # self.setWindowIcon(QtGui.QIcon(r"D:\Lgb\Self\favicon.ico"))       # 设置Icon的图标【公司】
        self.setGeometry(300, 300, 600, 400)
        # self.resize(300, 150)                                 # 设置窗口大小
        # self.move(300, 300)                                   # 设置位置
        self.center()                                         # 设置居中，见下面自定义函数。

        self.edit_text = QtGui.QLineEdit(self)
        self.edit_text.move(20, 30)
        self.edit_text.resize(460, 30)

        self.edit_text2 = QtGui.QLineEdit(self)
        self.edit_text2.move(20, 230)
        self.edit_text2.resize(460, 30)

        # ===================================== 按钮 =====================================
        self.okButton = QtGui.QPushButton(u"翻译一下", self)           # 设置按钮
        # self.okButton.mouseReleaseEvent = self.clickEvents           # 设置button的点击事件
        self.okButton.mouseReleaseEvent = self.fanyi_en2ch  # 设置button的点击事件
        self.okButton.move(500, 30)
        self.okButton.resize(80, 30)
        # self.okButton.setToolTip('This is a <b>QPushButton</b> widget')     # 设置按钮提示
        # self.okButton.resize(btn.sizeHint())                  # sizeHint()方法返回一个推荐的大小。
        # qbtn = QtGui.QPushButton('Quit', self)
        # qbtn.clicked.connect(QtCore.QCoreApplication.instance().quit)     # 按钮连接关闭的方法。关闭当前窗口
        # qbtn.clicked.connect(self.btnCloseEvent)
        # self.cancelButton = QtGui.QPushButton("Cancel")
        self.okButton2 = QtGui.QPushButton(u"翻译一下", self)           # 设置按钮
        # self.okButton.mouseReleaseEvent = self.clickEvents           # 设置button的点击事件
        self.okButton2.mouseReleaseEvent = self.fanyi_ch2en  # 设置button的点击事件
        self.okButton2.move(500, 230)
        self.okButton2.resize(80, 30)

        # ===================================== 标签 =====================================
        self.labelEn2CH = QtGui.QLabel(u'[ 英文 ] 翻译为 [ 中文 ]', self)
        self.labelEn2CH.move(20,10)
        # self.labels.setGeometry(50, 50, 150, 50)
        # self.labels.mouseReleaseEvent = self.events
        self.labelCH2En = QtGui.QLabel(u'[ 中文 ] 翻译为 [ 英文 ]', self)
        self.labelCH2En.move(20,200)

        # ===================================== 文本框 ===================================
        self.En2CH = QtGui.QTextEdit(u'  ', self)
        self.En2CH.setReadOnly(True)
        self.En2CH.resize(560,100)
        self.En2CH.move(20,70)

        self.CH2En = QtGui.QTextEdit(u'  ', self)
        self.CH2En.setReadOnly(True)
        self.CH2En.resize(560,100)
        self.CH2En.move(20,270)

        # 布局，未看。
        # hbox = QtGui.QHBoxLayout()
        # hbox.addStretch(1)
        # hbox.addWidget(self.okButton)
        # # hbox.addWidget(self.cancelButton)
        # vbox = QtGui.QVBoxLayout()
        # # vbox.addStretch(1)
        # vbox.addLayout(hbox)
        # self.setLayout(vbox)

    def fanyi_en2ch(self, event):
        q = str(self.edit_text.text())
        fromLang = 'en'  # 从..语言翻译
        toLang = 'zh'  # 翻译到..语言
        self.fanyi(fromLang, toLang, q)

    def fanyi_ch2en(self, event):
        q = str(self.edit_text2.text())
        fromLang = 'zh'  # 从..语言翻译
        toLang = 'en'  # 翻译到..语言
        self.fanyi(fromLang, toLang, q)

    def fanyi(self, fromLang, toLang, q):
        # q = 'apple'  # 翻译语句
        salt = random.randint(32768, 65536)  # 随机数
        sign = appid + q + str(salt) + secretKey  # sign
        m1 = md5.new()
        m1.update(sign)
        sign = m1.hexdigest()  # md5
        myurl = myurls + '?appid=' + appid + '&q=' + urllib.quote(q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(salt) + '&sign=' + sign
        try:
            httpClient = httplib.HTTPConnection('api.fanyi.baidu.com')
            httpClient.request('GET', myurl)
            response = httpClient.getresponse()
            response_dict2 = response.read()
            response_dict2 = json.loads(response_dict2)
            print response_dict2
            print response_dict2["trans_result"][0]["dst"]
            result = response_dict2["trans_result"][0]["dst"]
            # uni_result = result.decode("unicode-escape")
            if fromLang == 'en':
                # print "en"
                self.fanyi_show_en2ch(result)
            else:
                # print "ch"
                self.fanyi_show_ch2en(result)
        except Exception, e:
            print e
        finally:
            if httpClient:
                httpClient.close()

    def fanyi_show_ch2en(self, content):
        self.CH2En.setText(content)
        # self.En2CH.adjustSize()
        self.CH2En.update()

    def fanyi_show_en2ch(self, content):
        self.En2CH.setText(content)
        # self.En2CH.adjustSize()
        self.En2CH.update()

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

    def clickEvents(self, event):
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
        reply = QtGui.QMessageBox.question(self, u'警告', u"你确定要退出函函翻译吗?", QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No)
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


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    fanyi = fanyiApp()
    fanyi.show()
    sys.exit(app.exec_())

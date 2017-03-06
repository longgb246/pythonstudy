# #-*- coding:utf-8 -*-
# from __future__ import division
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import wx
#
#
# # 第一个简单的
# from matplotlib.backends import backend_wxagg
# from matplotlib.figure import Figure
# class TestFrame(wx.Frame):
#     def __init__(self):
#         # super(TestFrame, self).__init__(self, None)
#         wx.Frame.__init__(self, None)
#         self.panel = backend_wxagg.FigureCanvasAgg(self, -1, Figure())
#         axes = self.panel.figure.gca()
#         axes.cla()
#         axes.plot([1,2,3],[1,2,3])
#         self.panel.draw()
#
#
# def TestFrame_run():
#     app = wx.App()
#     f = TestFrame()
#     f.Show(True)
#     app.MainLoop()


# if __name__ == '__main__':
#     TestFrame_run()
#     pass



import numpy as np
import wx
import matplotlib

# matplotlib采用WXAgg为后台,将matplotlib嵌入wxPython中
matplotlib.use("WXAgg")
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_wx import NavigationToolbar2Wx as NavigationToolbar

######################################################################################
class MPL_Panel(wx.Panel):
    ''''' #MPL_Panel面板,可以继承或者创建实例'''
    def __init__(self,parent):
        wx.Panel.__init__(self,parent=parent, id=-1)

        self.Figure = matplotlib.figure.Figure(figsize=(4,3))
        self.axes = self.Figure.add_axes([0.1,0.1,0.8,0.8])
        self.FigureCanvas = FigureCanvas(self,-1,self.Figure)

        #继承鼠标移动显示鼠标处坐标的事件
        self.FigureCanvas.mpl_connect('motion_notify_event',self.MPLOnMouseMove)

        self.NavigationToolbar = NavigationToolbar(self.FigureCanvas)

        self.StaticText = wx.StaticText(self,-1,label='Show Help String')

        self.SubBoxSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.SubBoxSizer.Add(self.NavigationToolbar,proportion =0, border = 2,flag = wx.ALL | wx.EXPAND)
        self.SubBoxSizer.Add(self.StaticText,proportion =-1, border = 2,flag = wx.ALL | wx.EXPAND)

        self.TopBoxSizer = wx.BoxSizer(wx.VERTICAL)
        self.TopBoxSizer.Add(self.SubBoxSizer,proportion =-1, border = 2,flag = wx.ALL | wx.EXPAND)
        self.TopBoxSizer.Add(self.FigureCanvas,proportion =-10, border = 2,flag = wx.ALL | wx.EXPAND)

        self.SetSizer(self.TopBoxSizer)

    #显示坐标值
    def MPLOnMouseMove(self,event):

        ex=event.xdata#这个数据类型是numpy.float64
        ey=event.ydata#这个数据类型是numpy.float64
        if ex  and ey :
            #可以将numpy.float64类型转化为float类型,否则格式字符串可能会出错
            self.StaticText.SetLabel('%10.5f,%10.5f' % (float(ex),float(ey)))
            #也可以这样
            #self.StaticText.SetLabel('%s,%s' % (ex,ey))

###############################################################################
#  MPL_Frame添加了MPL_Panel的1个实例
###############################################################################
class MPL_Frame(wx.Frame):
    """MPL_Frame可以继承,并可修改,或者直接使用"""
    def __init__(self,title="MPL_Frame Example In wxPython",size=(800,500)):
        wx.Frame.__init__(self,parent=None,title = title,size=size)

        self.MPL = MPL_Panel(self)
        self.Figure = self.MPL.Figure
        self.axes = self.MPL.axes
        self.FigureCanvas = self.MPL.FigureCanvas

        self.RightPanel = wx.Panel(self,-1)
        #测试按钮1
        self.Button1 = wx.Button(self.RightPanel,-1,"TestButton",size=(100,40),pos=(10,10))
        self.Button1.Bind(wx.EVT_BUTTON,self.Button1Event)
        #创建FlexGridSizer
        self.FlexGridSizer=wx.FlexGridSizer( rows=5, cols=1, vgap=5,hgap=5)
        self.FlexGridSizer.SetFlexibleDirection(wx.BOTH)
        #加入Sizer中
        self.FlexGridSizer.Add(self.Button1,proportion =0, border = 5,flag = wx.ALL | wx.EXPAND)


        self.RightPanel.SetSizer(self.FlexGridSizer)

        self.BoxSizer=wx.BoxSizer(wx.HORIZONTAL)
        self.BoxSizer.Add(self.MPL,proportion =-10, border = 2,flag = wx.ALL | wx.EXPAND)
        self.BoxSizer.Add(self.RightPanel,proportion =0, border = 2,flag = wx.ALL | wx.EXPAND)

        self.SetSizer(self.BoxSizer)
        #MPL_Frame界面居中显示
        self.Centre(wx.BOTH)

    #按钮事件,用于测试绘图
    def Button1Event(self,event):
        x=np.arange(-10,10,0.25)
        y=np.cos(x)
        self.axes.plot(x,y,'--b*')
        self.axes.grid(True)
        self.FigureCanvas.draw()#一定要实时更新



########################################################################

#主程序测试
if __name__ == '__main__':
    app = wx.PySimpleApp()
    #frame = MPL2_Frame()
    frame =MPL_Frame()
    frame.Center()
    frame.Show()
    app.MainLoop()
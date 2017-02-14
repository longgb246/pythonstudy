#-*- coding:utf-8 -*-
########################################
#Author: wxuping
#Date: 2013-01-18
#Function: Show Image using wxpython,as well as matplotlib
#######################################
import numpy
import wx
from matplotlib.pyplot import imread


###########################################################################
## Class wxImagePanel
###########################################################################

class wxImagePanel ( wx.Panel ):
    def __init__( self, parent,wximage,size = wx.Size( 500,300 ) ):
        wx.Panel.__init__ ( self, parent, id = wx.ID_ANY, size = size )
        self.bSizer = wx.FlexGridSizer(rows=1, cols=1, vgap=10, hgap=10 )
        Width,Height=self.GetClientSize()
        self.wxbitmap=wx.BitmapFromImage(wximage.Rescale(Width,Height))
        self.m_bitmap1 = wx.StaticBitmap( self, wx.ID_ANY,self.wxbitmap, wx.DefaultPosition, wx.DefaultSize, 0 )
        self.bSizer.Add( self.m_bitmap1 )
        self.SetSizerAndFit( self.bSizer )
        self.Layout()



def SetwxImageData(numpyarray):
    """
    #本程序将numpyarray的数据加载到wxImage中
    #numpyarray一般从文件中读取
    #>>from matplotlib.pyplot import imread
    #>>numpyarray=imread(filename)
    #numpyarray必须是2维灰度图像或3维数组(R,G,B)
    #本程序的作用是可以将修改后的numpyarray加载到image中
    """
    import array
    nasize=len(numpyarray.shape)#测试数组的位数
    if nasize==2:
        Height,Width=numpyarray.shape
        wximage=wx.EmptyImage(Width,Height)
        Data=numpy.empty([Height,Width,3],dtype='byte')
        Data[:,:,0]=numpyarray
        Data[:,:,1]=numpyarray
        Data[:,:,2]=numpyarray
        wximage.SetData(numpy.flipud(Data).tostring())
    elif nasize==3:
        Height,Width,dim=numpyarray.shape
        wximage=wx.EmptyImage(Width,Height)
        if dim==3:#保存的是(R,G,B)
            #注意下面的代码,一定要将numpyarray上下反转一下
            wximage.SetData(numpy.flipud(numpyarray).tostring())
        elif dim==4:#某些格式,较少见
            Data=numpy.empty([Height,Width,3],dtype='byte')
            Data[:,:,:]=numpyarray[:,:,0:3]
            wximage.SetData(numpy.flipud(Data).tostring())
        elif dim==1:#其实仍然是灰度图像
            Data=numpy.empty([Height,Width,3],dtype='byte')
            Data[:,:,0]=numpyarray[:,:,0]
            Data[:,:,1]=numpyarray[:,:,0]
            Data[:,:,2]=numpyarray[:,:,0]
            wximage.SetData(numpy.flipud(Data).tostring())
    return wximage



class TestFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, size=(800,600))

        fgs = wx.GridSizer(cols=2, hgap=10, vgap=10)
        filename ="image.bmp"
        #1 直接读取文件并显示
        image1=wx.Image(filename,wx.BITMAP_TYPE_ANY)
        Width=image1.GetWidth()
        Height=image1.GetHeight()
        self.SetTitle("width:"+str(Width)+"   Height"+str(Height))
        self.wip1=wxImagePanel(self,image1,size=(600,500))

        #2 使用matplotlib读取图像文件并显示
        numpyarray=imread(filename)
        #注意numpyarray的第一维表示高度,第二维表示宽度
        image2=SetwxImageData(numpyarray)
        self.wip2=wxImagePanel(self,image2)

        # and put them into the sizer
        fgs.Add(self.wip1)
        fgs.Add(self.wip2)

        self.SetSizerAndFit(fgs)
        self.Center()


if __name__ == '__main__':
    app = wx.PySimpleApp()
    frm = TestFrame()
    frm.Show()
    app.MainLoop()
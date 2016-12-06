# coding: utf-8
import platform
import os.path
import sys
import time
windows_data_dir        ='D:/tmp/simulatePrograme/'
windows_output_dir      = 'simulation_results/'
# windows_logging_file    = ''


linux_data_dir          = '/home/cmo_ipc/stockPlan/data/'
linux_output_dir        = 'simulation_results/'
# linux_logging_file      = ''

date_range = ['2016-01-01', '2016-05-31']


pt                  = platform.system()
data_dir            = linux_data_dir
output_dir          = linux_output_dir
# logging_file        = linux_logging_file
if pt == 'Windows':
    data_dir        = windows_data_dir
    output_dir      = windows_output_dir
    # logging_file    = windows_logging_file

strategy={
    0:'SkuSimulation',
    1:'MaxVlt_Times_Demand',
    2:'SkuSimulationBp25',
    3:'SkuSimulationMg',
    4:'SkuSimulationPbs',
    5:'SkuSimulationSalesCorrection',
}
def start_point(workingfolderName):
    # f=open("start.txt")
    # while True:
    #     l = f.readline()
    #     print l
    #     if len(l)==0:
    #         break
    for key,value in strategy.iteritems():
        print "{0}:  {1}".format(key,value)
    try:
        re=int(raw_input("Your choice is: "))
        if re==-1:
            return re
        data_dir0= data_dir
        dirFolder = data_dir0+strategy[re]+'/'
        workingfolder = data_dir0+strategy[re]+'/'+workingfolderName+'/'
        outputfolder= workingfolder+output_dir+"/"
        logfolder= workingfolder+'log/'

        if not os.path.exists(dirFolder):
            os.mkdir(dirFolder)
        if not os.path.exists(workingfolder):
            os.mkdir(workingfolder)
        if not os.path.exists(outputfolder):
            os.mkdir(outputfolder)
        if not os.path.exists(logfolder):
            os.mkdir(logfolder)

        return int(re)
    except:
        print sys.exc_info()[0],sys.exc_info()[1]
        print "Please select again"
        return start_point(workingfolderName)

def setDataFileName():
    while True:
        print "Please upload your data file into folder:("+data_dir+")"
        f = raw_input("Please input your file name: ")
        if os.path.isfile(data_dir+f):
            return f


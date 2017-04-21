## test
import time
import os


# print time.localtime(time.time())
# print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
# print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) > "2017-04-20 20:30:00"
# print time.localtime(time.time()) > 1


def executeRun():
    os.system('nohup /usr/local/anaconda2/bin/python app_ioa_iaa_stdpre.py 2017-03-14 > nohup21.out 2>&1 &')
    os.system('nohup /usr/local/anaconda2/bin/python app_ioa_iaa_stdpre.py 2017-03-15 > nohup22.out 2>&1 &')
    os.system('nohup /usr/local/anaconda2/bin/python app_ioa_iaa_stdpre.py 2017-03-16 > nohup23.out 2>&1 &')
    os.system('nohup /usr/local/anaconda2/bin/python app_ioa_iaa_stdpre.py 2017-03-17 > nohup24.out 2>&1 &')
    os.system('nohup /usr/local/anaconda2/bin/python app_ioa_iaa_stdpre.py 2017-03-18 > nohup25.out 2>&1 &')
    os.system('nohup /usr/local/anaconda2/bin/python app_ioa_iaa_stdpre.py 2017-03-19 > nohup26.out 2>&1 &')
    os.system('nohup /usr/local/anaconda2/bin/python app_ioa_iaa_stdpre.py 2017-03-20 > nohup27.out 2>&1 &')
    os.system('nohup /usr/local/anaconda2/bin/python app_ioa_iaa_stdpre.py 2017-03-21 > nohup28.out 2>&1 &')
    os.system('nohup /usr/local/anaconda2/bin/python app_ioa_iaa_stdpre.py 2017-03-22 > nohup29.out 2>&1 &')
    os.system('nohup /usr/local/anaconda2/bin/python app_ioa_iaa_stdpre.py 2017-03-23 > nohup30.out 2>&1 &')
    os.system('nohup /usr/local/anaconda2/bin/python app_ioa_iaa_stdpre.py 2017-03-24 > nohup31.out 2>&1 &')
    os.system('nohup /usr/local/anaconda2/bin/python app_ioa_iaa_stdpre.py 2017-03-25 > nohup32.out 2>&1 &')
    os.system('nohup /usr/local/anaconda2/bin/python app_ioa_iaa_stdpre.py 2017-03-26 > nohup33.out 2>&1 &')
    os.system('nohup /usr/local/anaconda2/bin/python app_ioa_iaa_stdpre.py 2017-03-27 > nohup34.out 2>&1 &')
    os.system('nohup /usr/local/anaconda2/bin/python app_ioa_iaa_stdpre.py 2017-03-28 > nohup35.out 2>&1 &')
    os.system('nohup /usr/local/anaconda2/bin/python app_ioa_iaa_stdpre.py 2017-03-29 > nohup36.out 2>&1 &')
    os.system('nohup /usr/local/anaconda2/bin/python app_ioa_iaa_stdpre.py 2017-03-30 > nohup37.out 2>&1 &')
    os.system('nohup /usr/local/anaconda2/bin/python app_ioa_iaa_stdpre.py 2017-03-31 > nohup38.out 2>&1 &')


def circleRun():
    if time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) > target_time:
        executeRun()
        time.sleep(60)
        exit()
    else:
        time.sleep(60)


target_time = "2017-04-20 20:30:00"

if __name__ == "__main__":
    while True:
        circleRun()

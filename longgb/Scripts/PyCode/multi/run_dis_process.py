#-*- coding:utf-8 -*-
import os
from dis_process import main
import logging


# CRITICAL > ERROR > WARNING > INFO > DEBUG > NOTSET
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s (%(filename)s) [line:%(lineno)d] [ %(levelname)s ] %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    # filename='myapp.log',
                    # filemode='w'
                    )

logger = logging.getLogger("main")


if __name__ == '__main__':
    print 'Start main : PID[{0}]'.format(os.getpid())
    result_list = main(logger)
    logger.info('result_list[0] : {0}'.format(result_list[0]))
    logger.info('result_list[0] : {0}'.format(result_list[1]))
    logger.info('result_list[0] : {0}'.format(result_list[2]))
    logger.info(len(result_list))
    print 'result_list[0] : ', result_list[0]
    print 'result_list[1] : ', result_list[1]
    print 'result_list[2] : ', result_list[2]
    print len(result_list)


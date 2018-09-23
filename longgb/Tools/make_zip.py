# -*- coding: utf-8 -*-
'''
@author: longguangbin
@contact: longguangbin@163.com

将线上运行代码打包
'''

import zipfile
import os


def make_zip(source_dir, output_filename, only_zip=[], not_zip=[]):
    '''
    Package all the files.
    :param only_zip: suffix, ['py']
    :param not_zip: suffix, ['pyc']
    '''
    zipf = zipfile.ZipFile(output_filename, 'w')
    pre_len = len(os.path.dirname(source_dir))
    for parent, dirnames, filenames in os.walk(source_dir):
        for filename in filenames:
            flag = True
            if (len(not_zip) != 0) and (filename.split('.')[1] in not_zip):
                flag = False
            if (len(only_zip) != 0) and (filename.split('.')[1] not in only_zip):
                flag = False
            if flag:
                pathfile = os.path.join(parent, filename)
                arcname = pathfile[pre_len:].strip(os.path.sep)   # relative path
                print arcname
                zipf.write(pathfile, arcname)
    zipf.close()


def ziplib(mode='compile'):
    '''
    Package the online code to run.
    '''
    dir = os.path.dirname(__file__)
    lib_path = os.path.join(dir, '../src')
    # set it as save file path
    zip_path = './src.zip'
    if mode == 'compile':
        zf = zipfile.PyZipFile(zip_path, mode='w')
        zf.debug = 3
        zf.writepy(lib_path)
        zf.close()
    elif mode == 'zip':
        make_zip(lib_path, zip_path, only_zip=['py'])


if __name__ == '__main__':
    mode = 'compile'            # 'compile': pyc, 'zip': py
    ziplib(mode=mode)


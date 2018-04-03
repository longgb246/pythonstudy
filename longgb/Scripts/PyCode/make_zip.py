# -*- coding: utf-8 -*-
import zipfile
import os


def ziplib():
    """
    将线上运行代码打包
    :return:
    """
    dir = os.path.dirname(__file__)
    lib_path = os.path.join(dir, './src_test')

    # set it as save file path
    zip_path = './src_test.zip'
    zf = zipfile.PyZipFile(zip_path, mode='w')

    try:
        zf.debug = 3
        # This method will compile the module.py into module.pyc
        zf.writepy(lib_path)
        return zip_path
    finally:
        zf.close()


ziplib()
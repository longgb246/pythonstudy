#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'penghongxiao'

import os
os.system("export PYSPARK_PYTHON=python2.7.5")

from subprocess import call

file_path = "pop_order_vender_pre.py"

ok = call(["spark-submit",
      "--master", "yarn-client",
      "--num-executors", "25",
      "--executor-memory", "5g",
      "--executor-cores", "4",
      "--driver-memory", "4g",
      "--queue", "root.bdp_jmart_cmo_ipc_union.bdp_jmart_ipc_formal",
      file_path])
print("return status：" + str(ok))
if ok == 0:
    print("-------Successfully------")
else:
    print("-----------Error---------")
    raise Exception('ERROR：' + str(ok))

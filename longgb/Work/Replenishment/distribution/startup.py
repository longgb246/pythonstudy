#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'gaoyun3'
from subprocess import call
driver_path= "./VLT_OPTIMIZATION.py"
ok = call(["spark-submit",
           "--master", "yarn-client",
           "--num-executors", "25",
           "--executor-memory", "20g",
           "--executor-cores", "8",
           "--driver-memory", "10g",
           driver_path])
print("return code:"+str(ok))
if ok == 0:
    print("------------------------------------------")
    print("-----------------Success!-----------------")
    print("------------------------------------------")
else:
    print("------------------------------------------")
    print("-------------Failed:Exceotion-------------")
    print("------------------------------------------")
    raise Exception('ERROR:' + str(ok))
# -*- coding:utf-8 -*-
"""
  Author  : 'longguangbin'
  Contact : longguangbin@163.com
  Date    : 2018/9/5
  Usage   : 用于解决
"""
import os

over_path = r'/Users/longguangbin/Work/some/src'
target_path = r'/Users/longguangbin/Work/Codes/y-saas-forecast/src'


def read_file(path, mode='lf'):
    with open(path, 'r') as f:
        if mode == 'crlf':
            content = f.read()
        else:
            content = f.read()
            content = content.replace('\r', '')
    return content


def save_file(path, content):
    with open(path, 'w') as f:
        f.write(content)


def main():
    pre_len = len(os.path.dirname(over_path + '/'))
    for parent, dirnames, filenames in os.walk(over_path):
        for filename in filenames:
            pathfile = os.path.join(parent, filename)
            arcname = pathfile[pre_len:].strip(os.path.sep)  # relative path
            over_file = over_path + os.sep + arcname
            target_file = target_path + os.sep + arcname
            over_content = read_file(over_file)
            target_content = read_file(target_file)
            if over_content != target_content:
                print('Change file : {0}'.format(arcname))
                save_file(target_file, over_content)


if __name__ == '__main__':
    main()

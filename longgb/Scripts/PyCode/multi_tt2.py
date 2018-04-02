#-*- coding:utf-8 -*-
from multiprocessing import Pool
from functools import partial
import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys


def tt(x, user):
    print '[ PID({0}) ] user:{1}'.format(os.getpid(), user)
    text_res = x.find_element_by_tag_name('a').get_attribute('innerHTML')
    return text_res


def main():
    user = '2'

    driver = webdriver.Firefox()
    driver.get('http://www.python.org')
    elem = driver.find_element_by_id('about')
    first = partial(tt, user=user)

    x_sample = [elem]*10

    p = Pool(3)
    results = p.map(first, x_sample)
    driver.close()
    print results


if __name__ == '__main__':
    main()


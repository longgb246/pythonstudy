# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/2/20
"""  
Usage Of 'test_normal.py' : 
"""

import numpy as np
from scipy.stats import norm

norm.ppf(0.95, 0, 1)

norm.ppf(0.6, 1, np.sqrt(2))
norm.ppf(0.6, 1.5, np.sqrt(1 * 0.25 + 1 + 1 * 0.25))

norm.ppf(0.95, 2.0, np.sqrt(0.25+1+0.25*4))

# -*- coding:utf-8 -*-
def example1():
    '''
    I、II 产品生产问题
    '''
    import pulp as lp
    # variables
    Variables = ['I', 'II']
    Object = [2, 3]
    Facility = [1, 2]
    Material1 = [4, 0]
    Material2 = [0, 4]
    # problem
    prob = lp.LpProblem('The Problem 1', lp.LpMaximize)
    prob_vars = lp.LpVariable.dicts('Prob', Variables, 0)
    # object
    prob += lp.lpSum([Object[i] * prob_vars[Variables[i]] for i in range(len(Variables))])
    # subject
    prob += lp.lpSum([Facility[i] * prob_vars[Variables[i]] for i in range(len(Variables))]) <= 8
    prob += lp.lpSum([Material1[i] * prob_vars[Variables[i]] for i in range(len(Variables))]) <= 16
    prob += lp.lpSum([Material2[i] * prob_vars[Variables[i]] for i in range(len(Variables))]) <= 12
    print(prob)
    # solve
    prob.solve()
    print("Status:", lp.LpStatus[prob.status])
    result = {}
    for v in prob.variables():
        result[v.name] = v.varValue
        print(v.name, "=", v.varValue)
    print
    "Total Cost of Ingredients per can = ", lp.value(prob.objective)
    # ('Prob_I', '=', 4.0)
    # ('Prob_II', '=', 2.0)
    # Total Cost of Ingredients per can =  14.0


def example2():
    '''
    河流污染问题
    '''
    import pulp as lp
    # variables
    Variables = ['x1', 'x2']
    Object = [1000, 800]
    x1_limit = [1, 0]
    x2_limit = [0, 1]
    sub = [0.8, 1]
    # problem
    prob = lp.LpProblem('The Problem 2', lp.LpMinimize)
    # prob_vars = lp.LpVariable.dicts('Prob', Variables)
    prob_vars = lp.LpVariable.matrix('Prob', Variables)
    print
    prob_vars
    # object
    prob += lp.lpDot(Object, prob_vars)
    # subject
    prob += lp.lpDot(x1_limit, prob_vars) >= 1
    prob += lp.lpDot(x1_limit, prob_vars) <= 2
    prob += lp.lpDot(x2_limit, prob_vars) >= 0
    prob += lp.lpDot(x2_limit, prob_vars) <= 1.4
    prob += lp.lpDot(sub, prob_vars) >= 1.6
    print
    prob
    # solve
    prob.solve()
    print
    'Status', lp.LpStatus[prob.status]
    result = []
    for v in prob.variables():
        result.append([v.name, v.varValue])
        print
        v.name, '=', v.varValue
    print
    'The Optimal value is : ', lp.value(prob.objective)
    # Prob_x1 = 1.0
    # Prob_x2 = 0.8
    # The Optimal value is :  1640.0


# 单纯形法
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def simplexMethod(Object_list, Subject_list, init_bases=[], it=100):
    '''
    单纯形法：有一个问题，当系数都为负数时，但是截距却为负数，怎么办？（已达结束条件，但是约束：非负，不满足。）
    :param Object_list:
    :param Subject_list:
    :param init_bases:
    :param it:
    :return:
    '''

    def outputSimplexMethod(Object_list_len, bases, res, no_base_delta, object_v):
        '''
        Arrange the out format.
        '''
        X_vector = []
        index_str_list = no_base_delta.tolist()[0]
        base_i = 0
        no_base_i = 0
        func_str = 'z = {0} '.format(str(object_v))
        for i in range(Object_list_len):
            if base_i < len(bases) and bases[base_i] == i:
                X_vector.append(res[base_i])
                base_i += 1
            else:
                func_str += '{0} * x{1} '.format(str(index_str_list[no_base_i]), i)
                no_base_i += 1
                X_vector.append(0)
        func_str += ' \n[attention]: x{i} index i begin from 0.'
        return X_vector, func_str, object_v

    Object_list_len = len(Object_list)
    Object_m = np.matrix(Object_list)
    Subject_m = np.matrix(Subject_list)
    all_var = range(Subject_m.shape[1] - 1)
    if init_bases == []:
        len_m = int(Subject_m.shape[0])
        bases = range(len_m)
    else:
        bases = init_bases
    no_bases = list(set(all_var) - set(bases))
    object_v = None
    res = None
    no_base_delta = None
    while (it >= 0):
        it -= 1
        base_sub_m = Subject_m[:, bases]  # 取出基矩阵 m*m
        Subject_m = base_sub_m.I * Subject_m  # 计算新的subject m*n
        base_obj_m = Object_m[:, bases]  # 根据推导公式计算，z值
        z = base_obj_m * Subject_m
        delta = Object_m - z[:, :-1]  # 计算delta值，即为替换后系数
        no_base_delta = delta[:, no_bases]
        object_v = (Object_m[:, bases] * Subject_m[:, -1]).tolist()[0][0]
        res = (Subject_m[:, -1]).T.tolist()[0]
        if np.max(no_base_delta) <= 0:  # 迭代结束条件是系数都为负
            it = -1
        else:
            no_base_delta_index = np.argmax(no_base_delta)  # 正系数最大的被替入
            in_base_index = no_bases[no_base_delta_index]
            out_base_index = np.argmin(map(lambda x: x[0] if x[0] > 0 else np.inf,
                                           (Subject_m[:, -1] / Subject_m[:, in_base_index]).tolist()))  # 比例系数正最低的被替出
            bases = bases[:out_base_index] + bases[(out_base_index + 1):]
            bases = bases + [in_base_index]
            bases = sorted(bases)
            no_bases = list(set(all_var) - set(bases))
    print
    Subject_m
    X_vector, func_str, object_v = outputSimplexMethod(Object_list_len, bases, res, no_base_delta, object_v)
    return X_vector, func_str, object_v


# P30
Object_list = [2, 3, 0, 0, 0]
Subject_list = [[1, 2, 1, 0, 0, 8],
                [4, 0, 0, 1, 0, 16],
                [0, 4, 0, 0, 1, 12]]
init_bases = [2, 3, 4]
simplexMethod(Object_list, Subject_list, init_bases=init_bases, it=100)

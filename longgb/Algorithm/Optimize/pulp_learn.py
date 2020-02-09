# -*- coding:utf-8 -*-

# 一、简单的问题解决方案


def simpleSolution():
    import pulp as lp
    # Create the 'prob' variable to contain the problem data
    # 第一个参数：是这个问题的任意名称（作为字符串），第二个参数：是lp最小化或lp最大化
    prob = lp.LpProblem("The Whiskas Problem", lp.LpMinimize)
    # The 2 variables Beef and Chicken are created with a lower limit of zero
    # 它有四个参数：第一个是这个变量代表的任意名字，第二个是这个变量的下界，第三个是上界，第四是数据类型（离散的或连续的）。
    x1 = lp.LpVariable("ChickenPercent", 0, None, lp.LpInteger)
    # 第二、三参数：None表示无界，正or负无穷大。第四个参数：LpContinuous（默认）/LpInteger
    x2 = lp.LpVariable("BeefPercent", 0)
    # The objective function is added to 'prob' first
    # 目标函数是逻辑输入，在语句的末尾加上一个重要的逗号，以及一个简短的字符串来解释这个目标函数是什么
    prob += 0.013 * x1 + 0.008 * x2, "Total Cost of Ingredients per can"
    # The five constraints are entered                          #
    # 试试使用list，应该没有问题，逗号是tuple格式
    prob += x1 + x2 == 100, "PercentagesSum"  # 添加约束
    prob += 0.100 * x1 + 0.200 * x2 >= 8.0, "ProteinRequirement"
    prob += 0.080 * x1 + 0.100 * x2 >= 6.0, "FatRequirement"
    prob += 0.001 * x1 + 0.005 * x2 <= 2.0, "FibreRequirement"
    prob += 0.002 * x1 + 0.005 * x2 <= 0.4, "SaltRequirement"
    # The problem data is written to an .lp file                # writeLP
    # 将这些信息复制到一个.lp文件到代码块正在运行的目录中。
    prob.writeLP(r"D:\WhiskasModel.lp")
    # The problem is solved using PuLP's choice of Solver       #
    # 指定使用哪个求解程序。prob.solve(CPLEX())
    prob.solve()
    # The status of the solution is printed to the screen       #
    # 首先，我们要求解决方案的状态，它可以是“未解决”、“不可行的”、“无界”、“未定义”或“最优”的解决方案。
    # “Not Solved”, “Infeasible”, “Unbounded”, “Undefined” or “Optimal”
    print("Status:", lp.LpStatus[prob.status])
    # Each of the variables is printed with it's resolved optimum value
    for v in prob.variables():
        print(v.name, "=", v.varValue)
    # The optimised objective function value is printed to the screen
    print("Total Cost of Ingredients per can = ", lp.value(prob.objective))


# 二、完整的问题解决方案
def totalSolution():
    import pulp as lp
    # Creates a list of the Ingredients
    Ingredients = ['CHICKEN', 'BEEF', 'MUTTON', 'RICE', 'WHEAT', 'GEL']
    # A dictionary of the costs of each of the Ingredients is created
    costs = {'CHICKEN': 0.013,
             'BEEF': 0.008,
             'MUTTON': 0.010,
             'RICE': 0.002,
             'WHEAT': 0.005,
             'GEL': 0.001}
    # A dictionary of the protein percent in each of the Ingredients is created
    proteinPercent = {'CHICKEN': 0.100,
                      'BEEF': 0.200,
                      'MUTTON': 0.150,
                      'RICE': 0.000,
                      'WHEAT': 0.040,
                      'GEL': 0.000}
    # A dictionary of the fat percent in each of the Ingredients is created
    fatPercent = {'CHICKEN': 0.080,
                  'BEEF': 0.100,
                  'MUTTON': 0.110,
                  'RICE': 0.010,
                  'WHEAT': 0.010,
                  'GEL': 0.000}
    # A dictionary of the fibre percent in each of the Ingredients is created
    fibrePercent = {'CHICKEN': 0.001,
                    'BEEF': 0.005,
                    'MUTTON': 0.003,
                    'RICE': 0.100,
                    'WHEAT': 0.150,
                    'GEL': 0.000}
    # A dictionary of the salt percent in each of the Ingredients is created
    saltPercent = {'CHICKEN': 0.002,
                   'BEEF': 0.005,
                   'MUTTON': 0.007,
                   'RICE': 0.002,
                   'WHEAT': 0.008,
                   'GEL': 0.000}
    # Create the 'prob' variable to contain the problem data
    prob = lp.LpProblem("The Whiskas Problem", lp.LpMinimize)
    # A dictionary called 'ingredient_vars' is created to contain the
    # referenced Variables
    ingredient_vars = lp.LpVariable.dicts(
        "Ingr", Ingredients, 0)  # 变量名加了个前缀：Ingr_MUTTON，下限0
    # The objective function is added to 'prob' first
    prob += lp.lpSum([costs[i] * ingredient_vars[i]
                      for i in Ingredients]), "Total Cost of Ingredients per can"
    # The five constraints are added to 'prob'
    prob += lp.lpSum([ingredient_vars[i]
                      for i in Ingredients]) == 100, "PercentagesSum"
    prob += lp.lpSum([proteinPercent[i] * ingredient_vars[i]
                      for i in Ingredients]) >= 8.0, "ProteinRequirement"
    prob += lp.lpSum([fatPercent[i] * ingredient_vars[i]
                      for i in Ingredients]) >= 6.0, "FatRequirement"
    prob += lp.lpSum([fibrePercent[i] * ingredient_vars[i]
                      for i in Ingredients]) <= 2.0, "FibreRequirement"
    prob += lp.lpSum([saltPercent[i] * ingredient_vars[i]
                      for i in Ingredients]) <= 0.4, "SaltRequirement"
    prob.writeLP(r"D:\WhiskasModel2.lp")
    # The problem is solved using PuLP's choice of Solver       #
    # 指定使用哪个求解程序。prob.solve(CPLEX())
    prob.solve()
    # “Not Solved”, “Infeasible”, “Unbounded”, “Undefined” or “Optimal”
    print("Status:", lp.LpStatus[prob.status])
    # Each of the variables is printed with it's resolved optimum value
    for v in prob.variables():
        print
        v.name, "=", v.varValue
    # The optimised objective function value is printed to the screen
    print("Total Cost of Ingredients per can = ", lp.value(prob.objective))


# 三、数独问题
def shuDuSolution():
    import pulp as lp
    # A list of strings from "1" to "9" is created
    Sequence = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    # The Vals, Rows and Cols sequences all follow this form
    Vals = Sequence
    Rows = Sequence
    Cols = Sequence
    # The boxes list is created, with the row and column index of each square
    # in each box
    Boxes = []
    for i in range(3):
        for j in range(3):
            Boxes += [[(Rows[3 * i + k], Cols[3 * j + l])
                       for k in range(3) for l in range(3)]]
    # The prob variable is created to contain the problem data
    prob = lp.LpProblem("Sudoku Problem", lp.LpMinimize)
    # The problem variables are created
    choices = lp.LpVariable.dicts(
        "Choice", (Vals, Rows, Cols), 0, 1, lp.LpInteger)
    # The arbitrary objective function is added
    prob += 0, "Arbitrary Objective Function"
    # A constraint ensuring that only one value can be in each square is
    # created
    for r in Rows:
        for c in Cols:
            prob += lp.lpSum([choices[v][r][c] for v in Vals]) == 1, ""
    # The row, column and box constraints are added for each value
    for v in Vals:
        for r in Rows:
            prob += lp.lpSum([choices[v][r][c] for c in Cols]) == 1, ""
        for c in Cols:
            prob += lp.lpSum([choices[v][r][c] for r in Rows]) == 1, ""
        for b in Boxes:
            prob += lp.lpSum([choices[v][r][c] for (r, c) in b]) == 1, ""
    # The starting numbers are entered as constraints
    prob += choices["5"]["1"]["1"] == 1, ""
    prob += choices["6"]["2"]["1"] == 1, ""
    prob += choices["8"]["4"]["1"] == 1, ""
    prob += choices["4"]["5"]["1"] == 1, ""
    prob += choices["7"]["6"]["1"] == 1, ""
    prob += choices["3"]["1"]["2"] == 1, ""
    prob += choices["9"]["3"]["2"] == 1, ""
    prob += choices["6"]["7"]["2"] == 1, ""
    prob += choices["8"]["3"]["3"] == 1, ""
    prob += choices["1"]["2"]["4"] == 1, ""
    prob += choices["8"]["5"]["4"] == 1, ""
    prob += choices["4"]["8"]["4"] == 1, ""
    prob += choices["7"]["1"]["5"] == 1, ""
    prob += choices["9"]["2"]["5"] == 1, ""
    prob += choices["6"]["4"]["5"] == 1, ""
    prob += choices["2"]["6"]["5"] == 1, ""
    prob += choices["1"]["8"]["5"] == 1, ""
    prob += choices["8"]["9"]["5"] == 1, ""
    prob += choices["5"]["2"]["6"] == 1, ""
    prob += choices["3"]["5"]["6"] == 1, ""
    prob += choices["9"]["8"]["6"] == 1, ""
    prob += choices["2"]["7"]["7"] == 1, ""
    prob += choices["6"]["3"]["8"] == 1, ""
    prob += choices["8"]["7"]["8"] == 1, ""
    prob += choices["7"]["9"]["8"] == 1, ""
    prob += choices["3"]["4"]["9"] == 1, ""
    prob += choices["1"]["5"]["9"] == 1, ""
    prob += choices["6"]["6"]["9"] == 1, ""
    prob += choices["5"]["8"]["9"] == 1, ""
    # The problem data is written to an .lp file
    prob.writeLP("Sudoku.lp")
    # The problem is solved using PuLP's choice of Solver
    prob.solve()
    # The status of the solution is printed to the screen
    print("Status:", lp.LpStatus[prob.status])
    # A file called sudokuout.txt is created/overwritten for writing to
    sudokuout = open('sudokuout.txt', 'w')
    while True:
        prob.solve()
        # The status of the solution is printed to the screen
        print("Status:", lp.LpStatus[prob.status])
        # The solution is printed if it was deemed "optimal" i.e met the
        # constraints
        if lp.LpStatus[prob.status] == "Optimal":  # 是迭代器吗？居然可以边解边添加约束。
            # The solution is written to the sudokuout.txt file
            for r in Rows:
                if r == "1" or r == "4" or r == "7":
                    sudokuout.write("+-------+-------+-------+\n")
                for c in Cols:
                    for v in Vals:
                        if lp.value(choices[v][r][c]) == 1:
                            if c == "1" or c == "4" or c == "7":
                                sudokuout.write("| ")
                            sudokuout.write(v + " ")
                            if c == "9":
                                sudokuout.write("|\n")
            sudokuout.write("+-------+-------+-------+\n\n")
            # The constraint is added that the same solution cannot be returned again
            # 即为：这几个数加在一个不能还是81，否则便是同一解。
            prob += lp.lpSum([choices[v][r][c] for v in Vals
                              for r in Rows
                              for c in Cols
                              if lp.value(choices[v][r][c]) == 1]) <= 80
        # If a new optimal solution cannot be found, we end the program
        else:
            break
    sudokuout.close()
    # The location of the solutions is give to the user
    print("Solutions Written to sudokuout.txt")


# 四、一些函数
def someThing():
    import pulp
    pulp.combination([1, 2, 3, 4], 2)  # 2的组合
    pulp.allcombinations([1, 2, 3, 4], 2)  # 2以下组合
    pulp.permutation([1, 2, 3, 4], 2)  # 2的排列
    pulp.allpermutations([1, 2, 3, 4], 2)  # 2以下排列
    pass


# 五、线性优化问题
def linearMatrix():
    import pandas as pd
    import pulp as lp
    import numpy as np
    import random
    from pprint import pprint
    np.random.seed(1)
    row_num = 3
    column_num = 8
    raw_data = np.random.randint(100, size=(1, row_num * column_num))[0]
    print(raw_data)
    # A new LP problem
    prob = lp.LpProblem("Cannonical LP", lp.LpMinimize)
    a = list(range(row_num * column_num))
    print(a)
    b = list(map(lambda x: x % row_num == 3, a))
    print(b)
    x = lp.LpVariable.matrix(
        "x", list(range(row_num * column_num)), 0, 1, lp.LpInteger)
    print(x)
    # objective
    prob += lp.lpDot(raw_data, x)
    print(prob)
    print(lp.lpDot(x, list(map(lambda x: x % row_num == 0, a))))
    for count in range(column_num):
        prob += lp.lpSum(([x[i] for i in list(range(row_num *
                                                    column_num)) if i %
                           column_num == count])) == 1
        prob += lp.lpDot([x[j] for j in list(range(24)) if j %
                          column_num == count], [raw_data[j] for j in list(range(24)) if j %
                                                 column_num == count]) >= 1
    # #constrain
    # prob += lpSum(([x[i] for i in list(range(row_num * column_num)) if i % column_num == 0])) == 1
    # prob += lpSum(([x[i] for i in list(range(row_num * column_num)) if i % column_num == 2])) == 0
    # prob += lpSum(([x[i] for i in list(range(row_num * column_num)) if i % column_num == 3])) == 0
    # prob += lpSum(([x[i] for i in list(range(row_num * column_num)) if i % column_num == 4])) == 0
    # prob += lpSum(([x[i] for i in list(range(row_num * column_num)) if i % column_num == 5])) == 0
    # prob += lpSum(([x[i] for i in list(range(row_num * column_num)) if i % column_num == 6])) == 0
    # prob += lpSum(([x[i] for i in list(range(row_num * column_num)) if i % column_num == 7])) == 0
    # prob += lpSum(list(map(lambda x: x % 8 == 0, a))) >= 0.5
    # prob += lpDot(x, list(map(lambda x: x % 8 == 1, a))) >= 0.5
    # prob += lpDot(x, list(map(lambda x: x % 8 == 2, a))) >= 0.5
    # prob += lpDot(x, list(map(lambda x: x % 8 == 3, a))) >= 0.5
    # prob += lpDot(x, list(map(lambda x: x % 8 == 4, a))) >= 0.5
    # prob += lpDot(x, list(map(lambda x: x % 8 == 5, a))) >= 0.5
    # prob += lpDot(x, list(map(lambda x: x % 8 == 6, a))) >= 0.5
    # prob += lpDot(x, list(map(lambda x: x % 8 == 7, a))) >= 0.5
    # prob += lpDot(x, list(map(lambda x: x % 8 == 7, a))) <= 1
    # # prob += lpDot([raw_data[i] for i in list(range(24)) if i % 5 == 0], [x[i] for i in list(range(24)) if i % 5 == 0]) > 0
    # prob += lpDot([x[i] for i in list(range(24)) if i % 5 == 0], [raw_data[i] for i in list(range(24)) if i % 5 == 0]) >= 1
    # prob += lpDot([x[i] for i in list(range(24)) if i % 5 == 1], [raw_data[i] for i in list(range(24)) if i % 5 == 1]) >= 1
    # prob += lpDot([x[i] for i in list(range(24)) if i % 5 == 2], [raw_data[i] for i in list(range(24)) if i % 5 == 2]) >= 1
    # prob += lpDot([x[i] for i in list(range(24)) if i % 5 == 3], [raw_data[i] for i in list(range(24)) if i % 5 == 3]) >= 1
    # prob += lpDot([x[i] for i in list(range(24)) if i % 5 == 4], [raw_data[i] for i in list(range(24)) if i % 5 == 4]) >= 1
    # q = [raw_data[i] for i in list(range(24)) if i % 5 == 0]
    # w = [x[i] for i in list(range(24)) if i % 5 == 0]
    # prob += lpDot(q, w) < 5
    # print(lpDot(q, w))
    print(prob)
    # Resolution
    prob.solve()
    # Print the status of the solved LP
    print("Status:", lp.LpStatus[prob.status])
    result = {}
    for v in prob.variables():
        result[int(v.name[2:])] = [v.varValue]
        print(v.name, "=", v.varValue)
    # Print the value of the objective
    print("objective=", lp.value(prob.objective))
    varvale_list = [x.varValue for x in prob.variables()]
    # 需要排序
    print(sorted([int(x.name[2:]) for x in prob.variables()]))
    print(np.reshape(varvale_list, (row_num, column_num)))
    print(prob.variables())
    print(prob.variables()[1].value())
    print(result)
    result_list = []
    for i in range(row_num * column_num):
        result_list.append(result[i][0])
    print(result_list)
    print(np.reshape(result_list, (row_num, column_num)))
    print("Status:", lp.LpStatus[prob.status])


# 特征选取部分
# def feature_select(X,Y):
#     sel = VarianceThreshold()
#     tmp = set()
#     clf_rf = RandomForestRegressor(n_estimators=100,criterion='mae',max_depth=5)
#     clf_rf.fit(X,Y)
#     importance = clf_rf.feature_importances_
#     featrue_index = importance.argsort()[::-1]
#     for f in range(X.shape[1]):
#         if f < 100:                            #选出前100个重要的特征
#             tmp.add(X.columns[indices[f]])
#         print("%2d) %-*s %f" % (f + 1, 100, X.columns[indices[f]], importances[indices[f]]))
#         ##选取前50%
#     return tmp


# https://github.com/hyperopt/hyperopt-sklearn
# https://github.com/mabrek/kaggle-rossman-store-sales/blob/master/R/functions.R


# =========================================================================
# =                            python 魔法函数
# =========================================================================
class TT(object):
    def __init__(self):
        self.name = 'tt'

    def __str__(self):
        return self.name

    def __add__(self, other):
        return self.name + str(other)

    def __iadd__(self, other):
        self.name += other[0]
        # 万绮雯
        return self

# tt = TT()
# print tt
# print tt + 3
# tt += '55', 33
# print tt
# print tt.name

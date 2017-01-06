# coding=utf-8

import ast
import sys
import itertools
import pandas as pd
from subprocess import check_output


class WhiteListSolver:

    def __init__(self, parent_order_id, order_sku_record, fixed_sku, fixed_value, solver_path='', ind='', model_path='./',
                 model='model.lp', output='output.txt'):
        self.parent_order_id = parent_order_id
        self.order_sku_record = order_sku_record
        self.fixed_sku = fixed_sku
        self.fixed_value = fixed_value

        # 订单列表和SKU集合
        self.order_list = [ast.literal_eval(i) for i in self.order_sku_record]
        self.sku_list = list(set(list(itertools.chain(*self.order_list))))

        # SKU变量
        self.sku_var_index = [str(i + 1) for i in range(len(self.sku_list))]
        self.sku_var_dict = {i: 'x' + j for i, j in zip(self.sku_list, self.sku_var_index)}

        # Order变量
        self.ord_var_index = [str(i + 1) for i in range(len(self.parent_order_id))]
        self.ord_var_dict = {i: 'xi' + j for i, j in zip(self.parent_order_id, self.ord_var_index)}

        self.solver_path = solver_path
        self.model_path = model_path
        self.model = model
        self.output = output

        self.solution = None

    def construct_model(self, alpha=0.9):
        # 构建优化模型
        with open(self.model_path + str(ind)+self.model, 'a+') as f:

            f.write("min: ")

            # 目标函数
            f.write(' + '.join(self.sku_var_dict.values()) + ';\n')

            # 订单约束
            for x, xi in zip(self.order_list, self.parent_order_id):
                sku_var = [self.sku_var_dict[i] for i in x]
                sku_cnt = len(sku_var)
                f.write(' + '.join(sku_var) + ' - ' + str(sku_cnt) + ' ' + str(self.ord_var_dict[xi]) + ' >= 0' + ';\n')

            # 满足率约束
            right_hand = int(alpha * len(self.ord_var_dict))
            f.write(' + '.join(self.ord_var_dict.values()) + ' >= ' + str(right_hand) + ';\n')

            # 变量约束
            for var in self.sku_var_dict.values():
                f.write('0 <= ' + var + ' <= 1;\n')

            for var in self.ord_var_dict.values():
                f.write('0 <= ' + var + ' <= 1;\n')

            # 给定值的约束
            for sku_id, value in zip(self.fixed_sku, self.fixed_value):
                if sku_id in self.sku_var_dict:
                    f.write(self.sku_var_dict[sku_id] + ' = ' + str(value) + ';\n')

    def solve_model(self):
        # 调用命令行求解器
        if sys.platform.lower().startswith('win'):
            check_output(self.solver_path + "lp_solve.exe -s " + self.model_path + self.model + ' > ' +
                         self.model_path + self.output, shell=True)
        else:
            check_output(self.solver_path + "lp_solve -s " + self.model_path + self.model + ' > ' + self.model_path +
                         self.output, shell=True)

    def get_solution(self):
        # 解析模型输出
        with open(self.model_path + self.output, 'r') as f:
            lines = f.readlines()[4:]
        result = [line.strip().split() for line in lines]
        self.solution = {var: float(value) for var, value in result}

    def solution_to_file(self, path='', name='solution'+str(ind)+'.csv'):
        result = [[sku, self.sku_var_dict[sku], self.solution[self.sku_var_dict[sku]]] for sku in self.sku_list]
        result_df = pd.DataFrame.from_records(result)
        result_df.to_csv(path_or_buf=path + name, index=False, sep=',')

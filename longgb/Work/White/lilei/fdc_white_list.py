# coding: utf-8

import pandas as pd
import ast
import itertools
from subprocess import check_output

dtype = {'parent_sale_ord_id': str}
df = pd.read_csv('D:/white_list/white_list_history.csv', sep='\t', dtype=dtype)
parent_order_id = df['parent_sale_ord_id']
parent_order_id
order_sku = df['list_item_sku']
order_sku
# 订单列表和SKU集合
order_list = [ast.literal_eval(i) for i in order_sku]
sku_set = set(list(itertools.chain(*order_list)))
sku_list = list(sku_set)

# SKU变量
sku_var_index = [str(i + 1) for i in range(len(sku_list))]
sku_var_dict = {i: 'x' + j for i, j in zip(sku_list, sku_var_index)}

# Order变量
ord_var_index = [str(i + 1) for i in range(len(parent_order_id))]
ord_var_dict = {i: 'xi' + j for i, j in zip(parent_order_id, ord_var_index)}

# 构建优化模型
f = open('D:/white_list/model.lp', 'a+')

f.write("min: ")

# 目标函数
f.write(' + '.join(sku_var_dict.values()) + ';\n')

# 订单约束
for x, xi in zip(order_list, parent_order_id):
    sku_var = [sku_var_dict[i] for i in x]
    sku_cnt = len(sku_var)
    f.write(' + '.join(sku_var) + ' - ' + str(sku_cnt) + ' ' + str(ord_var_dict[xi]) + ' >= 0' + ';\n')

# 满足率约束
alpha = 0.9
right_hand = int(alpha * len(ord_var_dict))
f.write(' + '.join(ord_var_dict.values()) + ' >= ' + str(right_hand) + ';\n')

# 变量约束
for var in sku_var_dict.values():
    f.write('0 <= ' + var + ' <= 1;\n')

for var in ord_var_dict.values():
    f.write('0 <= ' + var + ' <= 1;\n')

f.close()

# 求解模型
check_output("lp_solve.exe -s model.lp > output.txt", shell=True)

# 解析模型输出
with open('D:/output.txt', 'r') as f:
    lines = f.readlines()[4:]

result = [line.strip().split() for line in lines]
solution = {var: float(value) for var, value in result}

# coding: utf-8

import pandas as pd
from WhiteListSolver import WhiteListSolver

dtype = {'parent_sale_ord_id': str}
df = pd.read_csv('E:/data/white_list_src_data.csv', sep='\t', dtype=dtype)
parent_order_id = df['parent_sale_ord_id']
order_sku_record = df['list_item_sku']

# 实例化求解器
model_path = 'D:/ipc_inv_opt/src/com/jd/test/chenchen/'
solver = WhiteListSolver(parent_order_id, order_sku_record, model_path=model_path)

solver.construct_model()
solver.solve_model()
solver.get_solution()
solver.solution_to_file()

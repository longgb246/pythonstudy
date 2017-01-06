# coding: utf-8

import pandas as pd
from WhiteListSolver import WhiteListSolver

for i in [i  for i in range(1, 13)]:
    ind = i
    dtype = {'parent_sale_ord_id': str}
    df = pd.read_csv('D:/white_list/white_list_history'+str(i)+'.csv', sep='\t', dtype=dtype)
    df_fixed = pd.read_csv('D:/white_list/wl_in_out'+str(i)+'.csv', sep=',', dtype={'sku_id': str})

    parent_order_id = df['parent_sale_ord_id']
    order_sku_record = df['list_item_sku']
    fixed_sku = df_fixed['sku_id']
    fixed_value = df_fixed['value']
    # 实例化求解器
    model_path = 'D:/white_list/'
    solver_path = 'D:/white_list/'
    solver = WhiteListSolver(parent_order_id, order_sku_record, fixed_sku, fixed_value, model_path=model_path,
                         solver_path=solver_path)
    solver.construct_model()
    solver.solve_model()
    solver.get_solution()
    solver.solution_to_file(path=model_path)


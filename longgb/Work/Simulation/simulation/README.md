---------------------2016-11-16 更新-------------------------------
@Author:Gaoyun
@Date: 2016-11-16
@Description: 新增交互功能，详细使用如下
1.  客户端（默认Windows）
     修改configServer.py中windows_data_dir、windows_output_dir（配置文件描述见第3点）
    1. cd run_simulation.py所在目录
    2. 运行 python cd run_simulation.py 见到如下界面：

    0:  SkuSimulation
    1:  HighVlt_SkuSimulation
    2:  SkuSimulationBp25
    3:  SkuSimulationMg
    4:  SkuSimulationPbs
    5:  SkuSimulationSalesCorrection
    6:  SkuSimulationSequential
    7:  HisSkuBpMeanSimulation

    3. 选择你需要使用的策略，例如：0，得到如下：
       Your choice is: 0
       Please upload your data file into folder:(D:/tmp/simulatePrograme/)
       Please input your file name:

    4. 把你的数据文件放到目录下，并输入你的文件全名
    5. 运行后程序自动创建如下目录，其中"2016-11-16_13-51-56" 为年月日小时分秒，保证不同用户可以同时运行同样的策略，并不会互相覆盖输出文件
       D:/tmp/simulatePrograme/SkuSimulation
       D:\tmp\simulatePrograme\SkuSimulation\2016-11-16_13-51-56
       D:\tmp\simulatePrograme\SkuSimulation\2016-11-16_13-51-56\simulation_results
       D:\tmp\simulatePrograme\SkuSimulation\2016-11-16_13-51-56\log

2.  服务器端（默认Linux）
       除了目录不同外其他操作同客户端。需要注意的是服务器端的工作目录为"/home/cmo_ipc/stockPlan/data"，把数据文件放在此目录下
       运行程序目录在"/home/cmo_ipc/stockPlan/ipc_inv_opt/src/com/jd/pbs/simulation"，运行服务器端步骤如下：
       cd /home/cmo_ipc/stockPlan/ipc_inv_opt/src/com/jd/pbs/simulation
       python2712   run_simulation.py
       之后的操作同客户端

3.  配置文件描述：configServer.py 内涵各个配置参数及交互方法
       configServer.py 参数描述：
       windows_data_dir/linux_data_dir：        自定义的工作的目录，此目录存放数据文件
       windows_output_dir/linux_output_dir：    输出文件夹名字。注： 程序最终输出目录为 windows_data_dir+ strategy（策略名称）+ YYYY-MM-DD_hh-mm-ss + windows_output_dir

       举例：windows_data_dir='D:/tmp/simulatePrograme/'。
       SkuSimulation策略的输出目录为 D:\tmp\simulatePrograme\SkuSimulation\2016-11-16_13-51-56\simulation_results
       SkuSimulation策略的输出log目录为 D:\tmp\simulatePrograme\SkuSimulation\2016-11-16_13-51-56\log

-------------------------------------------------------------------
# 仿真策略说明
1. 目前的仿真策略只考虑了补货，没有考虑内配、退货等业务对库存的影响。
2. 不进行仿真的几种情况：
- 仿真开始日期的销量预测数据为空。
- 仿真开始日期的库存数据为空。

## 仿真销量
1. 销量是指当天累计销量。
2. 使用历史真实销量作为仿真期间的需求。
3. 销量填充
   1. 有货平均销量小于1或者（有货且销量为0天数/有货天数）大于0.5时，随机抽取有货天的销量进行填充；
   2. 其它情况，使用有货天的销量中位数进行填充。

## 仿真库存
1. 仿真初始库存=仿真开始日期对应的历史库存。
2. 库存是指0点库存。

## 仿真采购单
1. 假设仿真过程中下发的采购单一定会足量送达。
2. 采购单对应的供应商送货时长从给定的VLT分布中随机抽取。

## 销量预测
1. 如果当天销量预测数据为空，使用前一天的数据填充。

## 补货建议数据
1. 如果当天补货建议数据为空，使用前一天的数据填充。

## 目标库存及补货量计算方式
1. `TI(Target Inventory) = LOP + BP*未来28天平均预测销量`
2. `补货量 = TI - 现货库存 - 在途`

## 仿真过程描述
1. 按天仿真，每日0时计算补货点，如果可用库存（现货库存+在途）低于补货点，触发补货。
2. 补货：计算采购量，从VLT分布中随机抽取一个VLT作为本次采购的供应商到货时长，下一天作为VLT的第一天。
3. **VLT抽取规则：本次采购的到货日期要大于或等于所有在途的到货日期。**

# 仿真程序说明
`SkuSimulation`是仿真基础类，主要包括以下几个函数：
1. `run_simulation()`: 按天仿真，调用`calc_lop()`判断是否需要补货，调用`replenishment()`进行补货。
2. `calc_lop()`: 计算补货点。
3. `replenishment()`: 执行补货，调用`calc_replenishment_quantity()`计算补货量，随机抽取VLT，更新采购在途、采购到货。
4. `calc_replenishment_quantity()`: 计算补货量。

## `SkuSimulation`中的策略说明
1. 补货点：定义随机变量`S=VLT期间的总需求`，其中VLT与每日需求都是随机的，使用正态分布近似计算该随机变量的分位点，即补货点。
2. 服务水平：默认使用PBS当天设置的CR。
3. BP：默认使用PBS当天设置的BP。

### 如何在使用`SkuSimulation`进行仿真时使用自定义的CR和BP？
可以在调用仿真方法时传入参数，例如`SkuSimulation.run_simulation(self, cr=0.95, bp=25)`。
**注意：目前只支持传入全局的CR和BP。**
如果要对每一天设置不同的CR和BP，可以通过继承`SkuSimulation`覆盖`calc_lop()`和`calc_replenishment_quantity()`实现。

### 如何在使用`SkuSimulation`进行仿真时保证两次仿真随机抽取的VLT序列一致？
在调用`run_simulation()`时传入参数`seed`，`SkuSimulation.run_simulation(self, seed=618)` 。

## 如何实现新的策略？
继承`SkuSimulation`类，重写`calc_lop()`与`calc_replenishment_quantity()`。
`SkuSimulationModify.py`中给出了几个例子。

### `SkuSimulationModify.py`中实现的策略说明
1. `SkuSimulationPbs`：使用Pbs线上策略给出的补货点和目标库存。
- 线上数据存在(补货点>现货库存+在途库存)但是(目标库存<现货库存+在途库存)的情况，目前的处理规则是不进行补货。
2. `SkuSimulationMg`：使用混合高斯分布的CDF精确计算补货点，使用Pbs线上设置的CR和BP。
3. `SkuSimulationBp25`：BP使用真实采购单BP的均值，大于25的截断，使用该类仿真时必须指定默认BP，`SkuSimulationBp25.run_simulation(self, bp=25)`。

## 如何修改仿真逻辑或者添加新的分析功能？
不要直接修改仿真基础类`SkuSimulation`!
与实现新策略一样，采用继承、重写的方式。

## 如何运行仿真程序？
简单来说，就是以下三步：
1. 读入数据
2. 实例化仿真类
3. 执行`run_simulation()`

仿真后的结果都作为成员变量存储在仿真对象中，供后续分析，后面会将一些常用的分析固化到`SkuSimulation`中。
`run_simulation_sample.py`是运行仿真的样例程序。
**注意：不要修改`run_simulation_sample.py`。**

### 关于`run_simulation_sample.py`的说明
1. 配置文件`config.py`
```
data_dir = 'E:/data/'
data_file_name = 'data_file_name.txt'
output_dir = 'E:/simulation_results/'
logging_dir = 'E:/simulation_results/'
```
2. 输入文件格式：字段含义参考表`dev.dev_pbs_inv_opt_sku_fact`的字段说明。

3. 仿真配置：
- `write_daily_data = False`，是否将仿真后的明细数据输出到文件，以csv格式
- `write_original_data = False`，是否将原始数据输出到文件，以csv格式
- `persistence = True`，是否持久化仿真结果，使用`pickle`
- `simulation_name = 'sample_data_base_policy'`，本次仿真的名字，用于命名输出文件

4. SKU粒度的KPI输出
   默认将SKU粒度的KPI结果输出到文件，以csv格式，输出字段包括：
- `sku_id`：商品ID
- `cr_sim`：仿真服务水平
- `cr_his` ：实际服务水平
- `ito_sim` ：仿真周转天数
- `ito_his`：实际周转天数
- `gmv_sim` ：仿真GMV
- `gmv_his` ：实际GMV
- `ts_sim` ：仿真总销量
- `ts_his` ：实际总销量
- `pur_cnt_sim` ：仿真采购次数（只统计仿真周期内到货的采购单）
- `success_cnt` ：仿真采购成功次数（到货当天库存大于零）
- `wh_qtn`：平均仓报价
- `pur_cnt_his`：实际采购次数
- `org_nation_sale_num_band`：全国销量BAND

5. 仿真结果（每日明细）有两种持久化方式：
6. 设置`write_daily_data = True`，将每个SKU的仿真明细以CSV格式输出到`output_dir`中
7. 默认方式：在`run_simulation_sample.py`中会创建一个字典`simulation_results`，字典的Key是`sku_id`，Value是仿真后的对象
- 该字典会被持久化到`output_dir`中（使用`pickle`），文件名为`simulation_name.dat`
- 该字典可以被load到Python程序中供后续分析使用，参考例子`load_sim_sample.py`

## `SkuSimulation`中提供的其它方法说明

### `SkuSimulation.get_daily_data()`

### `SkuSimulation.calc_kpi()`

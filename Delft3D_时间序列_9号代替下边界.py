# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 10:02:23 2020

相比于Delft3D_时间序列，使用9号的水质代替1号和10号的水质

@author: Carlisle
"""

import pandas as pd
from func import monitors, edit_bct_bcc, FlowGenerator
import sys
# reference_time = pd.to_datetime('2020-02-22 00:00:00')
# start, end = '2020-02-22 00:00:00', '2020-02-27 00:00:00'

# 外部调用传入参数，数据类型为字符串
reference_time = "{} {}".format(sys.argv[1], sys.argv[2])
reference_time = pd.to_datetime(reference_time)
start, end = "{} {}".format(sys.argv[3], sys.argv[4]), "{} {}".format(sys.argv[5], sys.argv[6])

bct_filename = 'river.bct'
bcc_filename = 'river.bcc'

yj_up = monitors(['12', '11'])
yj_down = monitors(['9'])
yj_6 = monitors(['6'])  # 用于构造污染物排放序列，6号水位低于1.0m时才排放污染物

# 污染物边界
up_dataset = yj_up.get_valid_and_resample_data(start, end, filter_type='2', stack=True)
yj_up.plot2('12')  # 检查数据有效性
yj_up.plot2('11')  # 检查数据有效性
up_transport_A = yj_up.completed_time_interpolate(up_dataset['12']['cond'], start, end, "10Min")
up_transport_B = yj_up.completed_time_interpolate(up_dataset['11']['cond'], start, end, "10Min")
up_transport_A = up_transport_A['cond'] / 1000  # 单位转换mg/l -> kg/m3
up_transport_B = up_transport_B['cond'] / 1000  # 单位转换mg/l -> kg/m3

down_dataset = yj_down.get_valid_and_resample_data(start, end, filter_type='2', stack=True)
yj_down.plot2('9')  # 检查数据有效性
# 由于1号和10号数据不全，所以两端都用9号的数据
down_transport_A = yj_down.completed_time_interpolate(down_dataset['9']['cond'], start, end, "10Min")
down_transport_B = yj_down.completed_time_interpolate(down_dataset['9']['cond'], start, end, "10Min")
down_transport_A = down_transport_A['cond'] / 1000  # 单位转换mg/l -> kg/m3
down_transport_B = down_transport_B['cond'] / 1000  # 单位转换mg/l -> kg/m3

# 上游水位边界
up_water_level = yj_up.completed_time_interpolate(up_dataset['12']['water level'], start, end, "10Min")
up_water_level = up_water_level['water level']

# 下游流量边界
# 一定要注意不要不要跨过2020-03-04 00:00:00
GetfLow = FlowGenerator()
down_flow = GetfLow.get_completed_flow(start, end)

# 生成构造污染物排放序列所需的流量和水位csv文件
# down_flow.to_csv("dflow/flow.csv")  # 生成流速csv，用于溯源过程中计算污染源排放时间的计算，退潮排放，涨潮不排放
# yj_6_data = yj_6.get_valid_and_resample_data(start, end, filter_type='2', stack=True)
# yj_6_data = yj_6.completed_time_interpolate(yj_6_data['6']['water level'], start, end, "10Min")
# yj_6_data['water level'].to_csv("dflow/yj6 water level.csv")

with open("dflow/" + bct_filename, 'r') as f:  # 读取bct文件
    bct_data = f.readlines()
with open("dflow/" + bcc_filename, 'r') as f:  # 读取bct文件
    bcc_data = f.readlines()

# 将datetime变成和referencetime的时间差，适应bct文件格式
bct_time = []
for i in up_water_level.index:
    delt = i - reference_time  # reference_time为参照时间
    delt_min = delt.total_seconds() / 60  # 秒换算成分钟
    bct_time.append(str("%e" % delt_min))

# 将时间和水位合在一起，加上换行符
bct_water_level = []
for i in range(len(up_water_level)):
    bct_water_level.append(' ' + bct_time[i] + ' ' + str("%e" % up_water_level[i]) \
                           + ' ' + str("%e" % up_water_level[i]) + '\n')

# 将时间和流量合在一起，加上换行符
bct_flow = []
for i in range(len(down_flow)):
    bct_flow.append(' ' + bct_time[i] + ' ' + str("%e" % down_flow[i]) \
                    + ' ' + str("%e" % down_flow[i]) + '\n')

bcc_up_transport = []
for i in range(len(up_transport_A)):
    bcc_up_transport.append(' ' + bct_time[i] + ' ' + str("%e" % up_transport_A[i]) \
                            + ' ' + str("%e" % up_transport_B[i]) + '\n')
bcc_down_transport = []
for i in range(len(down_transport_A)):
    bcc_down_transport.append(' ' + bct_time[i] + ' ' + str("%e" % down_transport_A[i]) \
                              + ' ' + str("%e" % down_transport_B[i]) + '\n')

# 将新的时间序列插入bct/bcc文件
bct_data = edit_bct_bcc(bct_data, reference_time, bct_water_level, bct_flow)  # 插入新数据
bcc_data = edit_bct_bcc(bcc_data, reference_time, bcc_up_transport, bcc_down_transport)  # 插入新数据

with open('dflow/Edit_' + bct_filename, 'w') as f:  # 写入bct文件
    for i in bct_data:
        f.write(i)

with open('dflow/Edit_' + bcc_filename, 'w') as f:  # 写入bcc文件
    for i in bcc_data:
        f.write(i)

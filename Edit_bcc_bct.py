# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 10:02:23 2020

@author: Carlisle
"""

import configparser
from Delft3DFiles import *
from func import monitors, FlowGenerator


# read config.ini
cf = configparser.ConfigParser()
cf.read('config.ini')
reference_time = cf.get('General', 'reference_time')
reference_time = pd.to_datetime(reference_time)
start, end = cf.get('General', 'start'), cf.get('General', 'end')
bct_filename = cf.get('Delft3D', 'bct_name')
bcc_filename = cf.get('Delft3D', 'bcc_name')

# -----------------------------------------------------------------------------------------------------------------
# preprocess time series data for modifying bct and bcc
# Please used your own data and preprocess methods to relace this part
yj_up = monitors(['12', '11'])
yj_down = monitors(['1', '10', '9'])

# upstream transport boundary
up_dataset = yj_up.get_valid_and_resample_data(start, end, filter_type='2', stack=True)
yj_up.plot2('12')  # check validity of data by plotting
yj_up.plot2('11')
up_transport_A = yj_up.completed_time_interpolate(up_dataset['12']['cond'], start, end, "10Min")
up_transport_B = yj_up.completed_time_interpolate(up_dataset['11']['cond'], start, end, "10Min")
up_transport_A = up_transport_A['cond'] / 1000  # mg/l -> kg/m3
up_transport_B = up_transport_B['cond'] / 1000  # mg/l -> kg/m3

# downstream transport boundary
down_dataset = yj_down.get_valid_and_resample_data(start, end, filter_type='2', stack=True)

# choose valid monitoring device
if len(down_dataset['1']['cond']) > 0 and len(down_dataset['10']['cond']) > 0:
    valid_device = ['1', '10']
elif len(down_dataset['1']['cond']) > 0 and len(down_dataset['10']['cond']) == 0:
    valid_device = ['1', '1']
elif len(down_dataset['1']['cond']) == 0 and len(down_dataset['10']['cond']) > 0:
    valid_device = ['10', '10']
else:
    valid_device = ['9', '9']

yj_down.plot2(valid_device[0])  # check validity of data by plotting
yj_down.plot2(valid_device[1])
down_transport_A = yj_down.completed_time_interpolate(down_dataset[valid_device[0]]['cond'], start, end, "10Min")
down_transport_B = yj_down.completed_time_interpolate(down_dataset[valid_device[1]]['cond'], start, end, "10Min")
down_transport_A = down_transport_A['cond'] / 1000  # mg/l -> kg/m3
down_transport_B = down_transport_B['cond'] / 1000  # mg/l -> kg/m3

# upstream water level boundary
up_water_level = yj_up.completed_time_interpolate(up_dataset['12']['water level'], start, end, "10Min")
up_water_level = up_water_level['water level']

# downstream flow boundary
# remember do not cross 2020-03-04 00:00:00
GetfLow = FlowGenerator()
down_flow = GetfLow.get_completed_flow(start, end)
# -----------------------------------------------------------------------------------------------------------------

# modified bcc and bct

# load bct and bcc file
bct = Delft3DTimeSeries("dflow/river.bct")
bcc = Delft3DTimeSeries("dflow/river.bcc")
# set the new time series
bct.set_time_series(0, reference_time, up_water_level, up_water_level)
bct.set_time_series(1, reference_time, down_flow, down_flow)
bcc.set_time_series(0, reference_time, up_transport_A, up_transport_B)
bcc.set_time_series(1, reference_time, down_transport_A, down_transport_B)
# write files
bct.to_file('dflow/Edited_river.bct')
bcc.to_file('dflow/Edited_river.bcc')


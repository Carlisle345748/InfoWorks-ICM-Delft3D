import os

import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from func import nse, edit_mdf_parm, monitors, bulk_insert


def run_bat(pm, obs_data, mdf_name, reference_time, start, end):
    if pm == -1:
        return 9999999999  # 若无效位置，则直接返回无效值
    else:
        # 修改模型
        edit_model(pm, mdf_name, reference_time, start, end)
        # 运行模型
        is_fail = os.system("cd dflow && run_{}.bat".format(pm))
        if is_fail:
            raise RuntimeError("模型运行失败")
        # 读取模拟数据
        YJ_ID = ['1', '10', '5', '9', '8', '6', '12', '11']  # 需读取数据的监测点id
        sim_water_level_pd, sim_cond_pd = read_result(pm, mdf_name, YJ_ID, reference_time)  # 从netcdf文件中读取数据
        # 删除文件
        delete_file(pm)
        # 模拟数据和监测数据结合，并检查数据完整性
        cond_data, water_level_data = combine_obs_sim(YJ_ID, start, end, obs_data, sim_cond_pd, sim_water_level_pd)
        # 计算目标函数值
        obj = get_obj(cond_data, water_level_data)
        return obj


def edit_model(pm, mdf_name, reference_time, start, end):
    """
    修改一系列模型文件
    :param pm: 参数
    :param mdf_name: mdf文件名
    :param reference_time: 参考时间
    :param start: 模拟开始时间
    :param end: 模拟结束时间
    :return: None
    """
    with open("dflow/" + mdf_name + ".mdf", "r") as file:
        data = file.readlines()

    edit_xml(pm, mdf_name)
    edit_bat(pm)
    edit_dis(pm, 'river', reference_time, start, end)
    data = edit_mdf_parm(data, 'Fildis', "river_{}.dis".format(pm))

    with open("dflow/{}_{}.mdf".format(mdf_name, pm), "w") as file:
        for lines in data:
            file.write(lines)


def edit_xml(pm, mdf_name):
    """
    生成新的xml文件
    :param pm: 命名随机数
    :param mdf_name: mdf文件名
    :return: None
    """
    with open("dflow/config_d_hydro.xml", "r") as file:
        data = file.readlines()
    # 修改模型名
    for index in range(len(data)):
        if data[index].find("        <mdfFile>") != -1:
            data[index] = "        <mdfFile>{}_{}.mdf</mdfFile>\n".format(mdf_name, pm)
        if data[index].find("urlFile") != -1:
            data[index] = "        <urlFile>{}_{}.url</urlFile>\n".format(mdf_name, pm)

    with open("dflow/config_d_hydro_" + str(pm) + ".xml", "w") as file:
        for each_line in data:
            file.write(each_line)


def edit_bat(pm):
    """
    生成新的bat文件
    :param pm: 命名随机数
    :return: None
    """
    with open("dflow/run.bat", "r") as file:
        data = file.readlines()
    for n in range(len(data)):
        if "config_d_hydro.xml" in data[n]:  # 修改config_d_hydro.xml
            data[n] = data[n].replace("config_d_hydro.xml", "config_d_hydro_{}.xml".format(pm))
            break
    else:
        raise RuntimeError('bat文件错误')  # 找不到config_d_hydro.xml则会触发错误
    with open("dflow/run_{}.bat".format(pm), "w") as file:
        for each_line in data:
            file.write(each_line)


def edit_dis(pm, dis_name, reference_time, start, end):
    """
    修改dis文件
    :param pm: 参数
    :param dis_name: dis文件名
    :param reference_time: 参照时间
    :param start: 模拟开始时间
    :param end: 模拟结束时间
    :return: None
    """
    with open("dflow/{}.dis".format(dis_name), 'r') as file:
        dis_data = file.readlines()
    # 参考时间格式转换
    reference_time = pd.to_datetime(reference_time)
    reference_time_str = reference_time.strftime("%Y%m%d")
    #  修改reference_time, location和 records-in-table
    time_range = pd.date_range(start, end, freq="10Min")
    start_index, end_index = [], []
    for ii in range(len(dis_data)):
        if dis_data[ii].find('reference-time') != -1:  # 修改reference_time
            dis_data[ii] = "reference-time       {}\n".format(reference_time_str)
        if dis_data[ii].find('records-in-table') != -1:  # 修改records-in-table
            dis_data[ii] = "records-in-table     {}\n".format(len(time_range))
            start_index.append(ii + 1)  # records-in-table下一行就是时间序列的开始
            n = ii + 1
            while n < len(dis_data) and 'table-name' not in dis_data[n]:  # 寻找时间序列的结尾
                n += 1
            end_index.append(n)

    # 插入污染物时间序列
    dis_data = insert_source_series(pm, dis_data, reference_time, start_index, end_index)
    # 写出新文件
    with open("dflow/{}_{}.dis".format(dis_name, pm), "w") as file:
        for lines in dis_data:
            file.write(lines)


def insert_source_series(pm, dis_data, reference_time, start_index, end_index):
    """
    往dis文件的list中删除旧时间序列，插入新时间序列
    :param dis_data: dis文件
    :param pm: 参数
    :param reference_time: 参照时间
    :param start_index: 旧时间序列开始index
    :param end_index: 旧时间序列结束index
    :return: 修改完的dis文件
    """
    pollutions = pd.read_csv("icm_to_delft3d/Link_{}_ds_cond.csv".format(pm))
    inflows = pd.read_csv("icm_to_delft3d/Link_{}_ds_flow.csv".format(pm))
    pollution_dict, inflow_dict = {}, {}
    for node in pollutions.columns[2:]:
        pollution_dict[node] = pollutions[node].copy()  # Delft3D中的单位也是kg/m3，因此不需要进行单位转换
        inflow_dict[node] = inflows[node].copy()  # Delft3D中的单位也是kg/m3，因此不需要进行单位转换
        inflow_dict[node].loc[inflow_dict[node] <= 0] = 0
    # 构造字符串时间序列的时间
    dis_time = []
    for ii in inflows['Time']:
        delta = pd.to_datetime(ii) - pd.to_datetime(reference_time)  # reference_time为参照时间
        delta_min = delta.total_seconds() / 60  # 秒换算成分钟
        dis_time.append(str("%e" % delta_min))
    if len(dis_time) != len(pollutions['yj3']):
        raise RuntimeError("从ICM导入的排放口污染物时间序列长度和Delft3D中设定的时间长度不一致")
    # 修改dis文件
    pretext_length = start_index[0]  # 时间序列前缀长度
    for index in range(end_index[0] - start_index[0]):  # 删除第一个时间序列
        del dis_data[pretext_length]
    for index in range(end_index[1] - start_index[1]):  # 删除第二个时间序列
        del dis_data[2 * pretext_length]
    for index in range(end_index[2] - start_index[2]):  # 删除第三个时间序列
        del dis_data[3 * pretext_length]
    for index in range(end_index[3] - start_index[3]):  # 删除第四个时间序列
        del dis_data[4 * pretext_length]

    # 将时间和水位合在一起，加上换行符
    yj3_source, yj4_source, yj5_source, yj6_source = [], [], [], []
    for ii in range(len(dis_time)):
        yj3_source.append(" {} {:e} {:e}\n".format(dis_time[ii], inflow_dict['yj3'][ii], pollution_dict['yj3'][ii]))
        yj4_source.append(" {} {:e} {:e}\n".format(dis_time[ii], inflow_dict['yj4'][ii], pollution_dict['yj4'][ii]))
        yj5_source.append(" {} {:e} {:e}\n".format(dis_time[ii], inflow_dict['yj5'][ii], pollution_dict['yj5'][ii]))
        yj6_source.append(" {} {:e} {:e}\n".format(dis_time[ii], inflow_dict['yj6'][ii], pollution_dict['yj6'][ii]))

    # 插入时间序列
    dis_data = bulk_insert(dis_data, pretext_length, yj3_source)
    dis_data = bulk_insert(dis_data, 2 * pretext_length + len(dis_time), yj4_source)  # 插入第二个时间序列
    dis_data = bulk_insert(dis_data, 3 * pretext_length + 2 * len(dis_time), yj5_source)  # 插入第二个时间序列
    dis_data = bulk_insert(dis_data, 4 * pretext_length + 3 * len(dis_time), yj6_source)  # 插入第二个时间序列
    return dis_data


def read_result(pm: int, mdf_name: str, yj_id: list, reference_time: str) -> tuple:
    """
    读取模型结果并整理
    :param reference_time:
    :param pm: 命名随机数
    :param mdf_name: 模型名
    :param yj_id: 需要提取的监测点id
    :return: 水位模拟值，污染物浓度模拟值
    """
    # 从netcdf文件中读取数据
    his = nc.Dataset("dflow/trih-{}_{}.nc".format(mdf_name, pm))  # 生成数据库
    timedelta = his.variables['time'][:]  # 读取相对时间
    sim_cond = his.variables['GRO'][:]  # 读取污染物浓度
    sim_water_level = his.variables['ZWL'][:]  # 读取水位
    his.close()  # 读取完一定要关闭数据库，不然下个循环会继续占用数据库文件，导致模型无法运行

    # 相对时间转换为绝对时间
    sim_time = pd.Series(np.ones(len(timedelta)))
    for index in range(len(timedelta)):
        sim_time[index] = pd.to_datetime(reference_time) + pd.to_timedelta(timedelta[index], unit='sec')

    # 将数据变成DataFrame，以时间为index，准备用于merge
    sim_cond_pd, sim_water_level_pd = {}, {}
    for index in range(sim_water_level.shape[1]):
        sim_cond_pd[yj_id[index]] = pd.Series(sim_cond[:, 0, 0, index], index=sim_time, dtype=np.float64, name='cond')
        sim_water_level_pd[yj_id[index]] = pd.Series(sim_water_level[:, index], index=sim_time,
                                                     dtype=np.float64, name='water level')
    return sim_water_level_pd, sim_cond_pd


def delete_file(pm):
    """
    删除临时模型文件，有时候会删除失败，所以用try来避免删除失败程序中断
    :param pm: 命名随机数
    :return: None
    """
    try:
        temp_files = os.listdir("dflow")
        for filename in temp_files:
            if filename.find(str(pm)) != -1:  # 删除含有命名随机数的文件
                os.remove("dflow/{}".format(filename))
        to_delft_files = os.listdir("icm_to_delft3d")
        for filename in to_delft_files:
            if filename.find(str(pm)) != -1:
                os.remove("icm_to_delft3d/{}".format(filename))
    except PermissionError:
        print('删除失败')
    return 0


def combine_obs_sim(YJ_ID, start, end, obs_data, sim_cond_pd, sim_water_level_pd):
    """
    结合模拟数据和监测数据在同一个表格里，并检查数据完整性
    :param YJ_ID: 监测点ID
    :param start: 模拟开始时间
    :param end: 模拟结束时间
    :param obs_data: 观测值
    :param sim_cond_pd: 污染物模拟值
    :param sim_water_level_pd: 水位模拟值
    :return: 污染物观测值+模拟值，水位观测值+模拟值
    """
    #
    completed_length = len(pd.date_range(start, end, freq="10Min"))
    cond_data = []
    for index in YJ_ID:
        if obs_data['cond'].get(index) is None or len(obs_data['cond'][index]) == 0:
            print("监测点{}没有有效污染物观测数据".format(index))
            continue
        elif len(sim_cond_pd[index]) < completed_length:  # 如果模拟数据长度小于completed_length，则模拟结果不收敛
            return 9999999999
        else:
            temp_data_1 = pd.merge(obs_data['cond'][index], sim_cond_pd[index], left_index=True, right_index=True)
            cond_data.append(temp_data_1)

    water_level_data = []
    for index in YJ_ID:  # 因为是理想模型，所以所有监测点都有水位
        if obs_data['water level'].get(index) is None or len(obs_data['water level'][index]) == 0:
            print("监测点{}没有有效水位观测数据".format(index))
            continue
        elif len(sim_water_level_pd[index]) < completed_length:  # 如果模拟数据长度小于completed_length，则模拟结果不收敛
            return 9999999999
        else:
            temp_data_2 = pd.merge(obs_data['water level'][index], sim_water_level_pd[index],
                                   left_index=True, right_index=True)
            water_level_data.append(temp_data_2)
    return cond_data, water_level_data


def get_obj(cond_data, water_level_data):
    """
    计算目标函数值
    :param cond_data: 污染物浓度的观测值和模拟值
    :param water_level_data: 水位的观测值和模拟值
    :return: 目标函数值
    """
    obj = 0  # 最小化目标值
    count = 0
    for obs_sim in cond_data:
        temp1 = obs_sim.values
        if np.sum(np.isinf(temp1)) != 0 or len(temp1) == 0:  # 如果存在无穷值或没有有效数据，则直接return9999999999，
            return 9999999999
        else:
            obj += nse(obs_sim.values)
            count += 1

    for obs_sim in water_level_data:
        temp2 = obs_sim.values
        if np.sum(np.isinf(temp2)) != 0 or len(temp2) == 0:  # 如果存在无穷值或没有有效数据，则直接return9999999999，
            return 9999999999
        else:
            obj += nse(obs_sim.values)
            count += 1

    obj = obj / count * -1  # *-1 转化为最小值问题，除以数据组数转化为最优解等于-1
    return obj


if __name__ == '__main__':
    MDF_NAME = 'river'
    R_TIME = "2020-03-31 00:00:00"  # 注意reference time总是00:00:00
    START, END = "2020-03-31 00:00:00", "2020-04-03 00:00:00"
    YJ_SET = ['1', '10', '5', '9', '8', '6', '12', '11']
    NETWORK = "4.7_model"
    OBS_FOLDER = 'MH52_0.5_1000.0'
    RUN_TEMPLATE = "3.31-4.03"

    # 运行脚本生成河流边界的时间序列，第一次运行需要运行这个！！！！！！
    os.system("C:/Users/Carlisle/Anaconda3/python.exe Delft3D_时间序列.py {} {} {}".format(R_TIME, START, END))

    # 读取确定河流模型初始条件所需的数据
    yj_initial = monitors(['1', '9', '10', '12'])
    yj_initial_data = yj_initial.get_valid_and_resample_data(START, END, filter_type='2', stack=True)

    # 读取观测值
    obs_his = nc.Dataset("obs/{}/trih-river.nc".format(OBS_FOLDER))  # 读取数据库
    time_delta = obs_his.variables['time'][:]  # 读取相对时间
    obs_cond = obs_his.variables['GRO'][:]  # 读取污染物浓度
    obs_water_level = obs_his.variables['ZWL'][:]  # 读取水位
    obs_his.close()  # 读取完一定要关闭数据库，不然下个循环会继续占用数据库文件，导致模型无法运行
    # 相对时间转换为绝对时间
    obs_time = pd.Series(np.ones(len(time_delta)))
    for i in range(len(time_delta)):
        obs_time[i] = pd.to_datetime(R_TIME) + pd.to_timedelta(time_delta[i], unit='sec')
    # 将数据变成DataFrame，以时间为index，准备用于merge
    obs_water_level_pd, obs_cond_pd = {}, {}
    for i in range(obs_water_level.shape[1]):
        obs_cond_pd[YJ_SET[i]] = pd.Series(obs_cond[:, 0, 0, i], index=obs_time, dtype=np.float64, name='cond')
        obs_water_level_pd[YJ_SET[i]] = pd.Series(obs_water_level[:, i], index=obs_time,
                                                  dtype=np.float64, name='water level')
    obs = {'water level': obs_water_level_pd, 'cond': obs_cond_pd}  # 整合观测值

    # 预先修改模型
    TSTART = (pd.to_datetime(START) - pd.to_datetime(R_TIME)).total_seconds() / 60  # 模拟开始时间
    TSTOP = (pd.to_datetime(END) - pd.to_datetime(R_TIME)).total_seconds() / 60  # 模拟结束时间
    # 初始水位和污染物浓度
    INITIAL_WATER_LEVEL = (yj_initial_data['1']['water level']['water level'][0] +
                           yj_initial_data['12']['water level']['water level'][0]) / 2  # 初始水位
    INITIAL_COND = (yj_initial_data['10']['cond']['cond'][0] + yj_initial_data['12']['cond']['cond'][0]) / 2 / 1000

    # 修改模型
    with open("dflow/river.mdf", 'r') as f:
        mdf_data = f.readlines()
        mdf_data = edit_mdf_parm(mdf_data, 'Itdate', pd.to_datetime(R_TIME).strftime("%Y-%m-%d"))  # 参考时间
        mdf_data = edit_mdf_parm(mdf_data, 'Tstart', TSTART)  # 模拟开始时间
        mdf_data = edit_mdf_parm(mdf_data, 'Tstop', TSTOP)  # 模拟结束时间
        mdf_data = edit_mdf_parm(mdf_data, 'Flmap', "{:e} 10  {:e}".format(TSTART, TSTOP), output=True)  # map文件时间
        mdf_data = edit_mdf_parm(mdf_data, 'Flhis', "{:e} 10  {:e}".format(TSTART, TSTOP), output=True)  # his文件时间
        mdf_data = edit_mdf_parm(mdf_data, 'Flpp', "{:e} 0  {:e}".format(TSTOP, TSTOP), output=True)  # rst文件时间
        mdf_data = edit_mdf_parm(mdf_data, 'C01', INITIAL_COND)  # 初始污染物浓度
        mdf_data = edit_mdf_parm(mdf_data, 'Zeta0', INITIAL_WATER_LEVEL)  # 初始水位
        mdf_data = edit_mdf_parm(mdf_data, 'Ccofu', 2.929357769349644203e-02)  # u方向摩擦系数
        mdf_data = edit_mdf_parm(mdf_data, 'Ccofv', 3.331360877592268177e-02)  # v方向摩擦系数
        mdf_data = edit_mdf_parm(mdf_data, 'Vicouv', 2.429591317843766163e+01)  # u方向涡流粘滞系数
        mdf_data = edit_mdf_parm(mdf_data, 'Dicouv', 2.005919323085833383e+01)  # v方向涡流扩散系数

    with open("dflow/river.mdf", 'w') as f:
        for line in mdf_data:
            f.write(line)
    # 运行算法
    bounds = [(505465, 506557), (2496786, 2497709), (0, 1), (300, 2000)]
    result = differential_evolution(run_bat, start=START, end=END, network=NETWORK,
                                    run_template=RUN_TEMPLATE, obs_folder=OBS_FOLDER,
                                    bounds=bounds, updating='deferred', workers=30, tol=0.0001,
                                    args=(obs, MDF_NAME, R_TIME, START, END), disp=True)
    print(result)

# start, end, reference_time = START, END, R_TIME
# pm = 462291884
# mdf_name = 'river'
# run_bat(pm, obs, 'river', R_TIME, START, END)
# YJ_ID = ['1', '10', '5', '9', '4', '8', '3', '6', '12', '11', '7']
# cond_data[1].plot()

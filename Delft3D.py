import os
import shutil
import numpy as np
import pandas as pd
import netCDF4 as nc
from func import nse, edit_mdf_parm, monitors, bulk_insert


class Delft3D(object):
    def __init__(self, reference_time, start, end, mdf_name,
                 dis_name, src_name, obs_folder=None):
        self.r_time = reference_time
        self.start, self.end = start, end
        self.mdf_name = mdf_name
        self.dis_name = dis_name
        self.src_name = src_name
        self.obs_folder = obs_folder
        self.yj_set = ['1', '10', '5', '9', '8', '6', '12', '11']

        self.initiate_mdf()
        self.initial_level, self.initial_cond = self.initial_state()
        if obs_folder is not None:
            self.obs = self.get_obs()

    def initial_state(self):
        yj_initial = monitors(['1', '9', '10', '12'])
        yj_initial_data = yj_initial.get_valid_and_resample_data(
            self.start, self.end, filter_type='2', stack=True)
        water_level = (yj_initial_data['1']['water level']['water level'][0] +
                       yj_initial_data['12']['water level']['water level'][0]) / 2  # 初始水位
        try:
            cond = (yj_initial_data['10']['cond']['cond'][0] +
                    yj_initial_data['12']['cond']['cond'][0]) / 2 / 1000  # 单位转换 mg/l -> kg/m3
        except KeyError:
            print("10号没有有效污染物数据，使用9号代替")
            cond = (yj_initial_data['9']['cond']['cond'][0] +
                    yj_initial_data['12']['cond']['cond'][0]) / 2 / 1000  # 单位转换 mg/l -> kg/m3
        return water_level, cond

    def initiate_mdf(self):
        self.initial_level, self.initial_cond = self.initial_state()
        Tstart = (pd.to_datetime(self.start) - pd.to_datetime(self.r_time)).total_seconds() / 60  # 模拟开始时间
        Tstop = (pd.to_datetime(self.end) - pd.to_datetime(self.r_time)).total_seconds() / 60  # 模拟结束时间

        with open("dflow/river.mdf", 'r') as f:
            mdf_data = f.readlines()
            # 参考时间
            mdf_data = edit_mdf_parm(mdf_data, 'Itdate', pd.to_datetime(self.r_time).strftime("%Y-%m-%d"))
            # 模拟开始时间
            mdf_data = edit_mdf_parm(mdf_data, 'Tstart', Tstart)
            # 模拟结束时间
            mdf_data = edit_mdf_parm(mdf_data, 'Tstop', Tstop)
            # map文件时间
            mdf_data = edit_mdf_parm(mdf_data, 'Flmap', "{:e} 10  {:e}".format(Tstart, Tstop), output=True)
            # his文件时间
            mdf_data = edit_mdf_parm(mdf_data, 'Flhis', "{:e} 10  {:e}".format(Tstart, Tstop), output=True)
            # rst文件时间
            mdf_data = edit_mdf_parm(mdf_data, 'Flpp', "{:e} 0  {:e}".format(Tstop, Tstop), output=True)
            # 初始污染物浓度
            mdf_data = edit_mdf_parm(mdf_data, 'C01', self.initial_cond)
            # 初始水位
            mdf_data = edit_mdf_parm(mdf_data, 'Zeta0', self.initial_level)
            # u方向摩擦系数
            mdf_data = edit_mdf_parm(mdf_data, 'Ccofu', 2.929357769349644203e-02)
            # v方向摩擦系数
            mdf_data = edit_mdf_parm(mdf_data, 'Ccofv', 3.331360877592268177e-02)
            # u方向涡流粘滞系数
            mdf_data = edit_mdf_parm(mdf_data, 'Vicouv', 2.429591317843766163e+01)
            # v方向涡流扩散系数
            mdf_data = edit_mdf_parm(mdf_data, 'Dicouv', 2.005919323085833383e+01)

        with open("dflow/river.mdf", 'w') as f:
            for line in mdf_data:
                f.write(line)

    def get_obs(self):
        obs_his = nc.Dataset("obs/{}/trih-river.nc".format(self.obs_folder))  # 读取数据库
        time_delta = obs_his.variables['time'][:]  # 读取相对时间
        obs_cond = obs_his.variables['GRO'][:]  # 读取污染物浓度
        obs_water_level = obs_his.variables['ZWL'][:]  # 读取水位
        obs_his.close()  # 读取完一定要关闭数据库，不然下个循环会继续占用数据库文件，导致模型无法运行
        # 相对时间转换为绝对时间
        obs_time = pd.Series(np.ones(len(time_delta)))
        for i in range(len(time_delta)):
            obs_time[i] = pd.to_datetime(self.r_time) + pd.to_timedelta(time_delta[i], unit='sec')
        # 将数据变成DataFrame，以时间为index，准备用于merge
        obs_water_level_pd, obs_cond_pd = {}, {}
        for i in range(obs_water_level.shape[1]):
            obs_cond_pd[self.yj_set[i]] = pd.Series(
                obs_cond[:, 0, 0, i], index=obs_time, dtype=np.float64, name='cond')
            obs_water_level_pd[self.yj_set[i]] = pd.Series(
                obs_water_level[:, i], index=obs_time, dtype=np.float64, name='water level')
        obs = {'water level': obs_water_level_pd, 'cond': obs_cond_pd}  # 整合观测值
        return obs

    def edit_model(self, pm):
        """
        修改一系列模型文件
        :param pm: 参数
        :return: None
        """
        with open("dflow/{}.mdf".format(self.mdf_name), "r") as file:
            data = file.readlines()

        self.edit_xml(pm)
        self.edit_bat(pm)
        self.edit_dis(pm)
        data = edit_mdf_parm(data, 'Fildis', "river_{}.dis".format(pm))

        with open("dflow/{}_{}.mdf".format(self.mdf_name, pm), "w") as file:
            for lines in data:
                file.write(lines)

    def edit_xml(self, pm):
        """
        生成新的xml文件
        :param pm: 命名随机数
        :return: None
        """
        with open("dflow/config_d_hydro.xml", "r") as file:
            data = file.readlines()
        # 修改模型名
        for index in range(len(data)):
            if data[index].find("        <mdfFile>") != -1:
                data[index] = \
                    "        <mdfFile>{}_{}.mdf</mdfFile>\n".format(self.mdf_name, pm)
            if data[index].find("urlFile") != -1:
                data[index] = \
                    "        <urlFile>{}_{}.url</urlFile>\n".format(self.mdf_name, pm)
        with open("dflow/config_d_hydro_" + str(pm) + ".xml", "w") as file:
            for each_line in data:
                file.write(each_line)

    @staticmethod
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
                data[n] = data[n].replace(
                    "config_d_hydro.xml", "config_d_hydro_{}.xml".format(pm))
                break
        else:
            # 找不到config_d_hydro.xml则会触发错误
            raise RuntimeError('bat文件错误')
        with open("dflow/run_{}.bat".format(pm), "w") as file:
            for each_line in data:
                file.write(each_line)

    def edit_dis(self, pm):
        """
        修改dis文件
        :param pm: 参数
        :return: None
        """
        with open("dflow/{}.dis".format(self.dis_name), 'r') as file:
            dis_data = file.readlines()
        # 参考时间格式转换
        reference_time = pd.to_datetime(self.r_time)
        reference_time_str = reference_time.strftime("%Y%m%d")
        #  修改reference_time, location和 records-in-table
        time_range = pd.date_range(self.start, self.end, freq="10Min")
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
        dis_data = self.insert_source_series(pm, dis_data, start_index, end_index)
        # 写出新文件
        with open("dflow/{}_{}.dis".format(self.dis_name, pm), "w") as file:
            for lines in dis_data:
                file.write(lines)

    def insert_source_series(self, pm, dis_data, start_index, end_index):
        """
        往dis文件的list中删除旧时间序列，插入新时间序列
        :param dis_data: dis文件
        :param pm: 参数
        :param start_index: 旧时间序列开始index
        :param end_index: 旧时间序列结束index
        :return: 修改完的dis文件
        """
        reference_time = pd.to_datetime(self.r_time)
        pollutions = pd.read_csv("icm_to_delft3d/Link_{}_ds_cond.csv".format(pm))
        inflows = pd.read_csv("icm_to_delft3d/Link_{}_ds_flow.csv".format(pm))
        pollution_dict, inflow_dict = {}, {}
        for node in pollutions.columns[2:]:
            # Delft3D中的污染物单位也是kg/m3，因此不需要进行单位转换
            pollution_dict[node] = pollutions[node].copy()
            inflow_dict[node] = inflows[node].copy()
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
            yj3_source.append(" {} {:e} {:e}\n".format(
                dis_time[ii], inflow_dict['yj3'][ii], pollution_dict['yj3'][ii]))

            yj4_source.append(" {} {:e} {:e}\n".format(
                dis_time[ii], inflow_dict['yj4'][ii], pollution_dict['yj4'][ii]))

            yj5_source.append(" {} {:e} {:e}\n".format(
                dis_time[ii], inflow_dict['yj5'][ii], pollution_dict['yj5'][ii]))

            yj6_source.append(" {} {:e} {:e}\n".format(
                dis_time[ii], inflow_dict['yj6'][ii], pollution_dict['yj6'][ii]))

        # 插入第一个时间序列
        dis_data = bulk_insert(dis_data, pretext_length, yj3_source)
        # 插入第二个时间序列
        dis_data = bulk_insert(dis_data, 2 * pretext_length + len(dis_time), yj4_source)
        # 插入第三个时间序列
        dis_data = bulk_insert(dis_data, 3 * pretext_length + 2 * len(dis_time), yj5_source)
        # 插入第四个时间序列
        dis_data = bulk_insert(dis_data, 4 * pretext_length + 3 * len(dis_time), yj6_source)
        return dis_data

    def read_result(self, pm: int) -> tuple:
        """
        读取模型结果并整理
        :param pm: 命名随机数
        :return: 水位模拟值，污染物浓度模拟值
        """
        # 从netcdf文件中读取数据
        his = nc.Dataset("dflow/trih-{}_{}.nc".format(self.mdf_name, pm))  # 生成数据库
        timedelta = his.variables['time'][:]  # 读取相对时间
        sim_cond = his.variables['GRO'][:]  # 读取污染物浓度
        sim_water_level = his.variables['ZWL'][:]  # 读取水位
        his.close()  # 读取完一定要关闭数据库，不然下个循环会继续占用数据库文件，导致模型无法运行

        # 相对时间转换为绝对时间
        sim_time = pd.Series(np.ones(len(timedelta)))
        for index in range(len(timedelta)):
            sim_time[index] = \
                pd.to_datetime(self.r_time) + pd.to_timedelta(timedelta[index], unit='sec')

        # 将数据变成DataFrame，以时间为index，准备用于merge
        sim_cond_pd, sim_water_level_pd = {}, {}
        for index in range(sim_water_level.shape[1]):
            sim_cond_pd[self.yj_set[index]] = pd.Series(
                sim_cond[:, 0, 0, index], index=sim_time, dtype=np.float64, name='cond')

            sim_water_level_pd[self.yj_set[index]] = pd.Series(
                sim_water_level[:, index], index=sim_time, dtype=np.float64, name='water level')

        return sim_water_level_pd, sim_cond_pd

    @staticmethod
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

    def combine_obs_sim(self, sim_cond_pd, sim_water_level_pd):
        """
        结合模拟数据和监测数据在同一个表格里，并检查数据完整性
        :param sim_cond_pd: 污染物模拟值
        :param sim_water_level_pd: 水位模拟值
        :return: 污染物观测值+模拟值，水位观测值+模拟值
        """
        # 完整数据长度
        completed_length = len(pd.date_range(self.start, self.end, freq="10Min"))

        # 合并污染物数据
        cond_data = []
        for index in self.yj_set:
            if self.obs['cond'].get(index) is None or len(self.obs['cond'][index]) == 0:
                print("监测点{}没有有效污染物观测数据".format(index))
                continue
            elif len(sim_cond_pd[index]) < completed_length:
                # 如果模拟数据长度小于completed_length，则模拟结果不收敛
                return 9999999999.0
            else:
                combined_data = pd.merge(self.obs['cond'][index],
                                         sim_cond_pd[index],
                                         left_index=True, right_index=True)
                cond_data.append(combined_data)

        # 合并水位数据
        water_level_data = []
        for index in self.yj_set:  # 因为是理想模型，所以所有监测点都有水位
            if self.obs['water level'].get(index) is None or\
                    len(self.obs['water level'][index]) == 0:
                print("监测点{}没有有效水位观测数据".format(index))
                continue
            elif len(sim_water_level_pd[index]) < completed_length:
                # 如果模拟数据长度小于completed_length，则模拟结果不收敛
                return 9999999999.0
            else:
                combined_data = pd.merge(self.obs['water level'][index],
                                         sim_water_level_pd[index],
                                         left_index=True, right_index=True)
                water_level_data.append(combined_data)

        return cond_data, water_level_data

    @staticmethod
    def get_obj(cond_data, water_level_data):
        """
        计算目标函数值
        :param cond_data: 污染物浓度的观测值和模拟值
        :param water_level_data: 水位的观测值和模拟值
        :return: 目标函数值
        """
        obj = 0  # 最小化目标值
        count = 0

        # 计算污染物数据nse
        for obs_sim in cond_data:
            temp1 = obs_sim.values
            if np.sum(np.isinf(temp1)) != 0 or len(temp1) == 0:
                # 如果存在无穷值或没有有效数据，则直接return9999999999
                return 9999999999.0
            else:
                obj += nse(obs_sim.values)
                count += 1

        # 计算水位数据nse
        for obs_sim in water_level_data:
            temp2 = obs_sim.values
            if np.sum(np.isinf(temp2)) != 0 or len(temp2) == 0:
                # 如果存在无穷值或没有有效数据，则直接return9999999999
                return 9999999999.0
            else:
                obj += nse(obs_sim.values)
                count += 1

        # 转化为最小值问题并维持最小值为-1
        obj = obj / count * -1
        return obj

    def solve(self, pm):
        """
        运行模型
        :param pm: ICM传入的污染物数据文件编号
        :return: 目标函数值
        """
        if pm == -1:
            return 9999999999.0  # 如果pm =1，则为无效位置，直接返回无效值
        else:
            # 修改模型
            self.edit_model(pm)
            # 运行模型
            is_fail = os.system("cd dflow && run_{}.bat".format(pm))
            if is_fail:
                raise RuntimeError("模型运行失败")
            # 读取模拟数据
            sim_water_level_pd, sim_cond_pd = self.read_result(pm)
            # 删除模型生成的文件
            self.delete_file(pm)
            # 模拟数据和监测数据结合，并检查数据完整性
            cond_data, water_level_data = self.combine_obs_sim(sim_cond_pd, sim_water_level_pd)
            # 计算目标函数值
            obj = self.get_obj(cond_data, water_level_data)

            return obj

    def create_obs(self, pm, population, all_valid_random_num, all_valid_locations):
        """运行Delft3D模型获取观测值"""
        # 修改模型
        self.edit_model(pm)
        # 运行模型
        is_fail = os.system("cd dflow && run_{}.bat".format(pm))
        if is_fail:
            raise RuntimeError("模型运行失败")
        # 重命名并移动结果文件
        self.move_delft3d_result(pm, all_valid_random_num,
                                 population, all_valid_locations)
        # 删除文件
        self.delete_file(pm)

    def move_delft3d_result(self, num, all_valid_random_num,
                            population, all_valid_locations):
        """寻找结果文件，重命名，并移动到obs下的对应子文件夹"""
        # 找到对应的参数
        index = np.argwhere(all_valid_random_num == num)[0, 0]
        member = population[index]
        location = all_valid_locations[index]
        flow, concentration = member[2:4]
        obs_folder = "obs/{}_{:.1f}_{:.1f}".format(location, flow, concentration)
        # 寻找对应的结果文件
        dflow_files = os.listdir("dflow")
        for file in dflow_files:
            # 重命名并移动文件
            if file.find("trih-{}_{}".format(self.mdf_name, num)) != -1:
                if not os.path.exists("{}/trih-river.nc".format(obs_folder)):
                    os.rename("dflow/{}".format(file), "dflow/trih-river.nc")
                    shutil.move("dflow/trih-river.nc", obs_folder)


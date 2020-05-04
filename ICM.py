import os
import time
import shutil
import numpy as np
import pandas as pd
import configparser
from matplotlib import pyplot as plt
from func import monitors, get_pollution_source, nse, bulk_insert


class InfoWorks(object):
    def __init__(self, population, config, plot=False):
        self.population = population
        self.parameter_num = population.shape[1] if len(population.shape) > 1 else len(population)
        self.population_num = population.shape[0] if len(population.shape) > 1 else 1
        self.file_random_num, self.random_num = self.get_random_num()
        self.valid_random_num = np.ones(self.population_num, dtype=np.int64) * int(-1)
        self.valid_location = [None] * self.population_num
        self.rb_command = None
        self.river_level = None
        self.plot = plot

        self.config = config
        cf = configparser.ConfigParser()
        cf.read(self.config)
        self.start = cf.get("General", "start")
        self.end = cf.get("General", "end")
        self.obs_folder = cf.get("General", "obs_folder")
        self.source_type = cf.get("General", "source_type")
        self.network = cf.get("InfoWorks", "network")  # 模型名
        self.run_template = cf.get("InfoWorks", "run_template")  # run模板名

        self.node_xy = pd.read_csv("icm_template/node_xy.csv", index_col=0)
        self.completed_length = len(pd.date_range(self.start, self.end, freq="10Min"))
        self.link_code = {'MH95.2': 'yj4', 'MH21.2': 'yj3',
                          'MH96.1': 'yj6', "MH132.4": 'yj5'}
        self.node_id = ['2', '17', '28', '14', '16', '15', '20', '22', '23',
                        '25', '26', '24', '19', '21', '30', '18', '27', '29']
        self.yj_code = dict(MH3='2', MH10='17', MH19='28', MH28='14', MH30='16',
                            MH44='15', MH45='20', MH50='22', MH52='23', MH60='25',
                            MH63='26', MH74='24', MH84='19', MH89='21', MH96='30',
                            MH103='18', MH117='27', MH121='29')
        self.obs = self.get_obs(self.obs_folder)

    def get_obs(self, folder):
        """获取观测值"""
        node_level = pd.read_csv("obs/{}/Node_level.csv".format(folder))
        node_cond = pd.read_csv("obs/{}/Node_cond.csv".format(folder))
        node_level.index = pd.to_datetime(node_level["Time"])
        node_level.index.name = 'time'
        node_cond.index = pd.to_datetime(node_cond["Time"])
        node_cond.index.name = 'time'
        colname = node_level.columns
        # 构建观测值数据集
        obs_water_level, obs_cond = {}, {}
        for name in colname[2:]:
            temp_level = node_level[name].copy()
            temp_level.name = 'water level'
            # 污染物记得要单位转换！！！kg/m3 -> mg/l
            temp_cond = node_cond[name].copy() * 1000
            temp_cond.name = 'cond'
            obs_water_level[self.yj_code[name]] = temp_level
            obs_cond[self.yj_code[name]] = temp_cond
        obs = {'water level': obs_water_level,
               'cond': obs_cond}  # 用字典保存，方便输入函数
        return obs

    @staticmethod
    def clean_up():
        """删除上次模拟生成的临时文件"""
        import os
        try:
            old_result_files = os.listdir('icm_result')
            old_model_files = os.listdir('icm_model_data')
            for filename in old_result_files:
                os.remove("icm_result/{}".format(filename))
            for filename in old_model_files:
                os.remove("icm_model_data/{}".format(filename))
        except PermissionError:
            print("删除失败，文件被占用")

    def get_random_num(self):
        """为种群的每个个体生成随机数"""
        file_random_num = np.random.randint(1, 1000000000, size=1)[0]  # network/run的随机数
        random_num = np.random.randint(1, 1000000000, size=self.population_num)  # 种群个体的随机数
        # 检验随机数是否重复
        is_duplicate = len(random_num) != len(set(random_num))
        while is_duplicate:
            random_num = np.random.randint(
                1, 1000000000, size=self.population_num)
            if len(random_num) == len(set(random_num)):
                break
        return file_random_num, random_num

    def check_location(self):
        """检查坐标是否有效"""
        count = 0
        for i in range(self.population_num):
            x, y = self.population[i, 0], self.population[i, 1]
            location = self.get_closest_node(x, y)
            self.valid_location[i] = location
            if location != 'invalid':
                # 标记为有效个体，对应的随机数有效(不等于-1)
                self.valid_random_num[i] = self.random_num[i]
                count += 1
            else:
                print("第{}个体位置无效".format(i+1))
        if count > 0:
            return True
        else:
            return False

    def get_closest_node(self, x, y):
        """获取坐标最近的节点的名字(未检验)"""
        dis = np.sqrt(
            (x - self.node_xy.values[:, 0]) ** 2 + (y - self.node_xy.values[:, 1]) ** 2)
        min_distance = np.min(dis)
        num = np.argmin(dis)
        if min_distance > 10:  # 与最近节点距离大于20m时，坐标无效
            return 'invalid'
        else:
            return self.node_xy.index[num]

    def get_level(self):
        """
        生成排放口处的水位曲线Delft3D.py
        """
        # 读取模板
        level = pd.read_csv('icm_template/level_template.csv', header=None)
        # 读取监测数据
        yj = monitors(['6'])  # 用6号的水位
        river = yj.get_valid_and_resample_data(self.start, self.end, stack=True)
        river_6 = yj.completed_time_interpolate(
            river['6']['water level'], self.start, self.end, "10Min")  # 6号监测点
        if self.plot:
            yj.plot2('6')
            plt.show()
        # 写入水位曲线
        if len(river_6['water level']) == self.completed_length:
            # 修改模板
            water_level = river_6['water level'].copy()
            water_level.iloc[0] = -0.1  # 让初始水位低于排放口，使管道内初始无水
            time_step = water_level.index[1] - water_level.index[0]
            time_step = int(time_step.total_seconds())
            for index, row in level.iterrows():
                if row[0] == "G_START" and row[1] == "G_TS":
                    # 修改时间
                    level.loc[index + 1, 1] = str(time_step)
                    level.loc[index + 1, 0] = \
                        water_level.index[0].strftime("%Y/%m/%d %H:%M")
                    break
            # 生成InfoWorks水位曲线
            temp_table = pd.DataFrame(
                {0: water_level.index, 1: water_level, 2: water_level, 3: water_level, 4: water_level, 5: water_level})
            river_level = pd.concat((level, temp_table), ignore_index=True)
            self.river_level = river_level
            # 导出水位曲线
            river_level.to_csv('icm_model_data/icm_level.csv', header=False, index=False)
        else:
            raise ValueError("6号水位数据不完整")

    def edit_nhd(self, NHD_data, location, river_cond, source_pollution, num):
        """为每个个体生成污染物曲线nhd文件"""
        NHD_data = NHD_data.copy()
        start = pd.to_datetime(self.start).strftime("%d%m%Y%H%M%S")  # 转换开始时间格式
        NHD_data[2] = NHD_data[2].replace("20012020000000", start, 1)  # 修改开始时间
        # 修改污染源位置
        NHD_data[8] = "{}".format(location.rjust(18)) + \
                      "       0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000\n"
        # 生成符合nhd格式的时间序列
        data = []
        for ii in range(len(river_cond)):
            data.append(
                "{:7.4f}{:7.4f}{:7.4f}{:7.4f}{:7.4f}{:7.4f}\n".format(river_cond[ii], river_cond[ii], river_cond[ii],
                                                                      river_cond[ii], river_cond[ii],
                                                                      source_pollution[ii]))
        NHD_data = bulk_insert(NHD_data, 9, data)  # 往nhd文件插入时间序列
        with open("icm_model_data/icm_pollution_{}.nhd".format(num), 'w') as f:
            for line in NHD_data:
                f.write(line)

    def get_pollutograph(self):
        """生成排放口和污染源的污染物曲线"""
        # 读取监测数据
        yj = monitors(['8'])
        river = yj.get_valid_and_resample_data(
            self.start, self.end, stack=True)
        river_8 = yj.completed_time_interpolate(
            river['8']['cond'], self.start, self.end, "10Min")
        river_cond = river_8['cond']
        if self.plot:
            yj.plot2('8')
            plt.show()
        if len(river_8['cond']) != self.completed_length:
            raise ValueError("8号污染物数据不完整")
        # 读取模板
        with open("icm_template/pollution_template.nhd", 'r') as f:
            NHD_data = f.readlines()
        # 为种群中的每个个体生成一份污染源配置文件
        for i in range(self.population_num):
            if self.valid_location[i] != 'invalid':
                if self.source_type == 'continue':
                    # 提取参数,生成污染源序列
                    flow, concentration = self.population[i, 2:4]
                    source_pollution = get_pollution_source(
                        self.start, self.end, [7, 18], concentration, flow)[1]
                elif self.source_type == 'instant':
                    # 提取参数,生成污染源序列
                    flow, concentration, start = self.population[i, 2:5]
                    source_pollution = self.get_instant_source(int(start), concentration, flow)[1]
                else:
                    raise ValueError("污染源类型输入错误")
                # 生成InfoWorks污染物曲线：排放口 + 污染源
                # 生成nhd文件
                self.edit_nhd(NHD_data, self.valid_location[i],
                              river_cond, source_pollution,
                              self.valid_random_num[i])

    def get_inflow(self):
        """生成污染物流量曲线"""
        # 读取模板
        inflow = pd.read_csv('icm_template/inflow_template.csv', header=None)
        # 修改模板
        time_step = 600  # 600秒
        for index, row in inflow.iterrows():
            if row[0] == "G_START" and row[1] == "G_TS":
                # 修改开始时间
                inflow.loc[index + 1, 0] = \
                    pd.to_datetime(self.start).strftime("%Y/%m/%d %H:%M")
                inflow.loc[index + 1, 1] = str(time_step)
                break
        # 生成每个个体的流量曲线
        for i in range(self.population_num):

            if self.valid_location[i] != 'invalid':
                # 生成污染物序列
                if self.source_type == 'continue':
                    flow, concentration = self.population[i, 2:4]
                    source_inflow = \
                        get_pollution_source(self.start, self.end, [7, 18], concentration, flow)[0]
                elif self.source_type == 'instant':
                    flow, concentration, start = self.population[i, 2:5]
                    source_inflow = self.get_instant_source(int(start), concentration, flow)[0]
                else:
                    raise ValueError("污染源类型输入错误")
                # 修改污染物位置
                inflow.iloc[8, 0] = self.valid_location[i]
                # 生成InfoWorks污染物入流曲线
                temp_table = pd.DataFrame(
                    {0: source_inflow.index, 1: source_inflow})
                pollution_inflow = pd.concat(
                    (inflow, temp_table), ignore_index=True)  # 与模板合并
                # 输出为配置文件
                pollution_inflow.to_csv("icm_model_data/icm_inflow_{}.csv".format(self.random_num[i]),
                                        header=False, index=False)

    def create_rb_command(self):
        """
        构建ruby命令，用于运行模型
        :return: None
        """
        cf = configparser.ConfigParser()
        cf.read(self.config)
        rb_command = cf.get("Path", 'icm_exchange')
        pwd = os.path.abspath('')
        ruby_script = cf.get("Path", 'ruby_script')
        pwd = os.path.join(pwd, ruby_script)
        rb_command = rb_command.replace('current_path', pwd)
        rb_command = rb_command.replace('random_num', str(self.file_random_num))
        rb_command = rb_command.replace('network', str(self.network))
        rb_command = rb_command.replace('run_template', str(self.run_template))
        self.rb_command = rb_command

    def read_sim_data(self, num: int) -> tuple:
        """读取模拟结果"""
        # 构建文件名
        level_file = "icm_result/Node_{}_DWF_depnod.csv".format(num)  # 水位结果文件名
        # 污染物结果文件名
        cond_file = "icm_result/Node_{}_DWF_mcnh4tot.csv".format(num)
        # 排放口流量结果文件名
        ds_flow_file = "icm_result/Link_{}_DWF_ds_flow.csv".format(num)
        # 排放口流量结果文件名
        ds_cond_file = "icm_result/Link_{}_DWF_ds_mcnh4tot.csv".format(num)
        # 读取文件
        level = pd.read_csv(level_file)
        cond = pd.read_csv(cond_file)
        ds_flow = pd.read_csv(ds_flow_file)
        ds_cond = pd.read_csv(ds_cond_file)
        # 将排放口的流量和污染物浓度写入csv文件，准备给Delft3d使用
        ds_flow.set_index('Time', inplace=True)
        ds_cond.set_index('Time', inplace=True)
        flow_old_cname = ds_cond.columns
        cond_old_cname = ds_flow.columns
        ds_flow.columns = ['seconds', self.link_code[flow_old_cname[1]], self.link_code[flow_old_cname[2]],
                           self.link_code[flow_old_cname[3]], self.link_code[flow_old_cname[4]]]
        ds_cond.columns = ['seconds', self.link_code[cond_old_cname[1]], self.link_code[cond_old_cname[2]],
                           self.link_code[cond_old_cname[3]], self.link_code[cond_old_cname[4]]]
        ds_flow.to_csv("icm_to_delft3d/Link_{}_ds_flow.csv".format(num))
        ds_cond.to_csv("icm_to_delft3d/Link_{}_ds_cond.csv".format(num))
        # 提取监测点水位和污染物浓度数据
        column_name = list(level.columns)
        time_index = pd.to_datetime(level['Time'])
        water_level_sim, cond_sim = {}, {}
        start_time = time_index[0] + pd.to_timedelta("1day")
        for cname in column_name[2:]:
            water_level_sim[self.yj_code[cname]] = pd.Series(
                level[cname].values, index=time_index, name='water level')
            water_level_sim[self.yj_code[cname]] = \
                water_level_sim[self.yj_code[cname]][start_time:]  # 去除第一天的数据
            # 污染物记得要单位转换！！！kg/m3 -> mg/l
            cond_sim[self.yj_code[cname]] = pd.Series(
                cond[cname].values * 1000, index=time_index, name='cond')
            cond_sim[self.yj_code[cname]] = \
                cond_sim[self.yj_code[cname]][start_time:]  # 去除第一天的数据
        return water_level_sim, cond_sim

    def get_nse(self, water_level_obs, cond_obs, water_level_sim, cond_sim):
        """
        根据观测值和模拟值计算nse，忽略不含有效数据的监测点
        Parameter
        ----------
        Yj_code: dict
        节点号和监测点号的转换
        water_level_obs: dict
        水位观测值
        cond_obs: dict
        污染物浓度观测值
        water_level_sim： dict
        水位模拟值
        cond_sim: dict
        污染物浓度模拟值
        Returns
        -------
        obj: float
        nse值
        """
        obj = 0  # 最小化目标值
        counts = 0  # 有效数据组数
        for yj in list(self.yj_code.values()):
            try:
                water_level_data = pd.merge(water_level_obs[yj], water_level_sim[yj],
                                            how='inner', left_index=True, right_index=True)
                cond_data = pd.merge(cond_obs[yj], cond_sim[yj],
                                     how='inner', left_index=True, right_index=True)
            except KeyError:
                print(yj + '无有效数据')
            else:
                if len(water_level_data) > 0:
                    obj += nse(water_level_data.values)
                    counts += 1
                if len(cond_data) > 0:
                    obj += nse(cond_data.values)
                    counts += 1
        return -1 * obj / counts

    def solve(self):
        """运行模拟"""

        if not self.check_location():  # 检验是否有效位置
            return np.full(self.population_num, 9999999999.0), self.valid_random_num
        else:
            self.clean_up()  # 清理旧的临时文件
            self.create_rb_command()  # 生成rb命令
            self.get_level()  # 生成水位文件
            self.get_pollutograph()  # 生成污染物曲线文件
            self.get_inflow()  # 生成入流文件
            # 保存随机数让ruby读取
            np.savetxt('icm_model_data/valid_random_num.csv',
                       self.valid_random_num, delimiter=",", fmt="%d")
            # 运行模型
            start_time = time.time()
            status = os.system(self.rb_command)  # 用ruby写入参数并运行模型
            trial_times = 0
            while status != 0 and trial_times < 20:
                time.sleep(5)
                status = os.system(self.rb_command)  # 重新运行模型
                trial_times += 1
            end_time = time.time()
            print("任务耗时:{}".format(end_time - start_time))
            expected_result_num = np.nonzero(self.valid_random_num != -1)[0].shape[0] * 4
            print("预期结果文件有{}个，实际结果文件共有{}个".format(
                expected_result_num, len(os.listdir("icm_result"))))

            # 读取结果
            obj = np.full(self.population_num, 9999999999.0)
            for n in range(self.population_num):
                if self.valid_random_num[n] != -1:
                    try:
                        # 读取模拟数据
                        water_level_sim, cond_sim = self.read_sim_data(self.random_num[n])
                        y = self.get_nse(self.obs['water level'], self.obs['cond'], water_level_sim, cond_sim)  # 计算nse
                    except FileNotFoundError:
                        raise FileExistsError("模型结果文件不存在")
                    else:
                        obj[n] = y

            return obj, self.valid_random_num

    def move_result(self):
        """
        在obs文件夹的生成子文件夹储存观测值重命名并移动结果文件到所生成的子文件夹
        """
        result_files = os.listdir("icm_result")  # 读取文件名
        for i in range(self.population_num):
            # 读取个体参数，构造文件名
            location = self.valid_location[i]
            if self.source_type == 'continue':
                flow, concentration = self.population[i, 2:4]
                obs_folder = "obs/{}_{:.1f}_{:.1f}".format(
                    location, flow, concentration)
            elif self.source_type == 'instant':
                flow, concentration, start = self.population[i, 2:5]
                obs_folder = "obs/{}_{:.1f}_{:.1f}_{:.1f}".format(
                    location, flow, concentration, start)
            else:
                raise ValueError("污染源类型输入错误")
            try:
                os.makedirs(obs_folder)
            except FileExistsError:
                continue
            # 转移并重命名文件
            num = self.valid_random_num[i]
            for file in result_files:
                if file.find(str(num)) != -1:
                    # 重命名不同种类的结果文件
                    if "Node" in file and "depnod" in file:
                        new_file_name = "Node_level.csv"
                    elif "Node" in file and "mcnh4tot" in file:
                        new_file_name = "Node_cond.csv"
                    elif "Link" in file and "ds_flow" in file:
                        new_file_name = "Link_ds_flow.csv"
                    elif "Link" in file and "ds_mcnh4tot" in file:
                        new_file_name = "Link_ds_cond.csv"
                    else:
                        raise RuntimeError("文件名格式错误")
                    os.rename("icm_result/{}".format(file),
                              "icm_result/{}".format(new_file_name))
                    shutil.move("icm_result/{}".format(new_file_name), obs_folder)

    def create_obs(self):
        """生成观测值"""
        self.check_location()
        self.clean_up()  # 清理旧的临时文件
        self.create_rb_command()  # 生成新的bat文件
        self.get_level()  # 生成水位文件
        self.get_pollutograph()  # 生成污染物曲线文件
        self.get_inflow()  # 生成污染物入流文件
        # 保存随机数，让ruby读取
        np.savetxt('temp/valid_random_num.csv', self.valid_random_num,
                   delimiter=",", fmt="%d")
        # 运行模型
        start_time = time.time()
        status = os.system(self.rb_command)  # 用ruby写入参数并运行模型
        trial_times = 0
        while status != 0 and trial_times < 20:
            time.sleep(5)
            status = os.system(self.rb_command)  # 重新运行模型
            trial_times += 1
        end_time = time.time()
        print("任务耗时:{}".format(end_time - start_time))
        expected_result_num = np.nonzero(self.valid_random_num != -1)[0].shape[0] * 4
        print("预期结果文件有{}个，实际结果文件共有{}个".format(
            expected_result_num, len(os.listdir("icm_result"))))

        for n in range(self.population_num):
            try:
                # 这个读取模拟数据的函数顺便把结果文件转移到icm_to_delft3d，以供delft3d运行使用
                self.read_sim_data(self.random_num[n])
            except FileNotFoundError:
                raise FileExistsError("模型结果文件不存在")

        # 转移结果文件到obs文件
        self.move_result()
        return self.valid_random_num, self.valid_location

    def get_instant_source(self, start_time, concentration, flow):
        """生成节点瞬时排放污染源时间序列"""
        time_range = pd.date_range(self.start, self.end, freq="10Min")  # 时间
        pollution_start = pd.to_datetime(self.start) + pd.to_timedelta(start_time*10, unit="Min")
        # 定义污染和入流时间序列
        pollution_source = pd.Series(np.zeros(len(time_range)), index=time_range)
        inflow = pd.Series(np.zeros(len(time_range)), index=time_range)
        pollution_source[pollution_start] = concentration
        inflow[pollution_start] = flow
        return inflow, pollution_source

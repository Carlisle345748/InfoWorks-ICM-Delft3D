import configparser
from multiprocessing.pool import Pool

from Delft3D import *
from ICM import *

# ----------------------------------------------------------------------------------------------------------------------
# 耦合测试
if __name__ == "__maim__":
    cf = configparser.ConfigParser()
    cf.read("config.ini")

    R_TIME = cf.get("General", "reference_time")  # 注意reference time总是00:00:00
    START = cf.get("General", "start")
    END = cf.get("General", "end")
    RUN_TEMPLATE = cf.get("InfoWorks", "run_template")
    SOURCE_TYPE = cf.get("General", "source_type")
    MDF_NAME = cf.get("Delft3D", "mdf_name")
    DIS_NAME = cf.get("Delft3D", "dis_name")
    SRC_NAME = cf.get("Delft3D", "src_name")
    NETWORK = cf.get("InfoWorks", "network")
    OBS_FOLDER = cf.get("General", "obs_folder")
    # 模拟参数

    # ICM模块测试
    # population = np.array([[505680.248, 2497248.282, 0.5, 1000],
    #                        [505680.248, 2497248.282, 0.5, 1000]])
    population = np.array([[505680.248, 2497248.282, 0.5, 1000, 144],
                           [505680.248, 2497248.282, 0.5, 1000, 144]])
    icm_test = InfoWorks(population, start=START, end=END,
                         network=NETWORK, run_template=RUN_TEMPLATE,
                         obs_folder=OBS_FOLDER, plot=False)

    icm_energies, valid_random_num = icm_test.solve()
    print(icm_energies)

    # Delft3D模块测试
    delft3d_test = Delft3D(reference_time=R_TIME, start=START, end=END,
                           mdf_name=MDF_NAME,  dis_name=DIS_NAME,
                           src_name=SRC_NAME, obs_folder=OBS_FOLDER)
    pool = Pool(processes=2)
    delft3d_energies = pool.map(delft3d_test.solve, valid_random_num)
    print(delft3d_energies)

# ----------------------------------------------------------------------------------------------------------------------
# 耦合测试
# class _FunctionWrapper(object):
#     """
#     Object to wrap user cost function, allowing picklability
#     """
#
#     def __init__(self, f, args):
#         self.f = f
#         self.args = [] if args is None else args
#
#     def __call__(self, x):
#         return self.f(x, *self.args)
#
#
# if __name__ == '__main__':
#     MDF_NAME = 'river'
#     R_TIME = "2020-03-31 00:00:00"  # 注意reference time总是00:00:00
#     START, END = "2020-03-31 00:00:00", "2020-04-03 00:00:00"
#     YJ_SET = ['1', '10', '5', '9', '8', '6', '12', '11']
#
#     obs_his = nc.Dataset("obs/trih-river.nc")  # 读取数据库
#     time_delta = obs_his.variables['time'][:]  # 读取相对时间
#     obs_cond = obs_his.variables['GRO'][:]  # 读取污染物浓度
#     obs_water_level = obs_his.variables['ZWL'][:]  # 读取水位
#     obs_his.close()  # 读取完一定要关闭数据库，不然下个循环会继续占用数据库文件，导致模型无法运行
#     # 相对时间转换为绝对时间
#     obs_time = pd.Series(np.ones(len(time_delta)))
#     for i in range(len(time_delta)):
#         obs_time[i] = pd.to_datetime(R_TIME) + pd.to_timedelta(time_delta[i], unit='sec')
#     # 将数据变成DataFrame，以时间为index，准备用于merge
#     obs_water_level_pd, obs_cond_pd = {}, {}
#     for i in range(obs_water_level.shape[1]):
#         obs_cond_pd[YJ_SET[i]] = pd.Series(obs_cond[:, 0, 0, i], index=obs_time, dtype=np.float64, name='cond')
#         obs_water_level_pd[YJ_SET[i]] = pd.Series(obs_water_level[:, i], index=obs_time,
#                                                   dtype=np.float64, name='water level')
#     obs = {'water level': obs_water_level_pd, 'cond': obs_cond_pd}  # 整合观测值
#
#     args = (obs, MDF_NAME, R_TIME, START, END)
#     func = _FunctionWrapper(run_bat, args)
#
#     population = np.array([[505680.248, 2497248.282, 0.5, 1000], [505680.248, 2497248.282, 0.5, 1000]])
#     test = InfoWorks(population, start="2020-03-31 00:00:00", end="2020-04-03 00:00:00",
#                      network="4.7_model", run_template="3.31-4.03",
#                      obs_folder='MH52_0.5_1000', plot=False)
#     icm_energies, valid_random_num = test.solve()
#     print(icm_energies)
#
#     pool = Pool(processes=2)
#     hh = pool.map(func, valid_random_num)
#     print(hh)

# ------------------------------------------------------------------------------------------------------------------------
# 调试单个结果文件
# population = np.array([[505680.248, 2497248.282, 0.5, 1000], [505680.248, 2497248.282, 0.5, 1000]])
# test = InfoWorks(population, "2020-03-31 00:00:00", "2020-04-03 00:00:00", "4.7_model", "3.31-4.03", plot=False)
# obs = test.get_obs()
#
# level_file = "icm_result/Node_{}_DWF_depnod.csv".format(651849552)  # 水位结果文件名
# cond_file = "icm_result/Node_{}_DWF_mcnh4tot.csv".format(651849552)
# level = pd.read_csv(level_file)
# cond = pd.read_csv(cond_file)
#
# column_name = list(level.columns)
# time_index = pd.to_datetime(level['Time'])
# water_level_sim, cond_sim = {}, {}
# start_time = time_index[0] + pd.to_timedelta("1day")
# for cname in column_name[2:]:
#     water_level_sim[test.yj_code[cname]] = pd.Series(
#         level[cname].values, index=time_index, name='water level')
#     water_level_sim[test.yj_code[cname]
#     ] = water_level_sim[test.yj_code[cname]][start_time:]  # 去除第一天的数据
#     # 污染物记得要单位转换！！！kg/m3 -> mg/l
#     cond_sim[test.yj_code[cname]] = pd.Series(
#         cond[cname].values * 1000, index=time_index, name='cond')
#     cond_sim[test.yj_code[cname]
#     ] = cond_sim[test.yj_code[cname]][start_time:]  # 去除第一天的数据

# obj = 0
# counts = 1
# for yj in test.yj_code.values():
#     water_level_data = pd.merge(obs['water level'][yj], water_level_sim[yj],
#                                 how='inner', left_index=True, right_index=True)
#     cond_data = pd.merge(obs['cond'][yj], cond_sim[yj],
#                          how='inner', left_index=True, right_index=True)
#     if len(water_level_data) > 0:
#         obj += nse(water_level_data.values)
#         print(nse(water_level_data.values))
#         counts += 1
#     if len(cond_data) > 0:
#         obj += nse(cond_data.values)
#         print(yj, nse(cond_data.values))
#         counts += 1
# obj = -1 * obj / counts

# ------------------------------------------------------------------------------------------------------------------------
# from func import RandomContainer
# from multiprocessing.pool import Pool
#
#
# class _FunctionWrapper(object):
#     """
#     Object to wrap user cost function, allowing picklability
#     """
#
#     def __init__(self, f, args):
#         self.f = f
#         self.args = [] if args is None else args
#
#     def __call__(self, x):
#         return self.f(x, *self.args)
#
#
# def tt(x, valid_random_num):
#     return x, next(valid_random_num)
#
#
# if __name__ == '__main__':
#     pool = Pool(processes=2)
#     valid = RandomContainer([1, 2, 3, 4])
#     func = _FunctionWrapper(tt, (valid,))
#     hh = pool.map(func, [1, 2, 3])
#     print(hh)

# from scipy.optimize import rosen, differential_evolution
# bounds = [(0, 2), (0, 2), (0, 2), (0, 2), (0, 2)]
# result = differential_evolution(rosen, bounds, updating='deferred')

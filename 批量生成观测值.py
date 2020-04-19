from ICM import *
from ICM_Delft3D import *
from func import FunctionWrapper
from multiprocessing.pool import Pool


def delft3d_obs(num, mdf_name, reference_time, start, end,
                all_valid_random_num, population, all_valid_locations):
    """运行Delft3D模型获取观测值"""
    # 修改模型
    edit_model(num, mdf_name, reference_time, start, end)
    # 运行模型
    is_fail = os.system("cd dflow && run_{}.bat".format(num))
    if is_fail:
        raise RuntimeError("模型运行失败")
    # 重命名并移动结果文件
    move_delft3d_result(num, mdf_name, all_valid_random_num,
                        population, all_valid_locations)
    # 删除文件
    delete_file(num)


def move_delft3d_result(num, mdf_name, all_valid_random_num,
                        population, all_valid_locations):
    """寻找结果文件，重命名，并移动到obs下的对应子文件夹"""
    # 找到对应的参数
    index = np.argwhere(all_valid_random_num == num)[0, 0]
    member = population[index]
    location = all_valid_locations[index]
    flow, concentration = member[2:4]
    obs_folder = "obs/{}_{}_{}".format(location, flow, concentration)
    # 寻找对应的结果文件
    dflow_files = os.listdir("dflow")
    for file in dflow_files:
        # 重命名并移动文件
        if file.find("trih-{}_{}".format(mdf_name, num)) != -1:
            os.rename("dflow/{}".format(file), "dflow/trih-river.nc")
            shutil.move("dflow/trih-river.nc", obs_folder)


if __name__ == "__main__":
    # 生成待测试条件
    # TEST = np.array([[505680.248, 2497248.282, 0.5, 1000],
    #                  [505680.248, 2497248.282, 0.5, 1000],
    #                  [505680.248, 2497248.282, 0.5, 1000],
    #                  [505680.248, 2497248.282, 0.5, 1000],
    #                  [505680.248, 2497248.282, 0.5, 1000],
    #                  [505680.248, 2497248.282, 0.5, 1000],
    #                  [505680.248, 2497248.282, 0.5, 1000]])
    CONCENTRATION = np.arange(100, 2100, 100).reshape(-1, 1)
    FLOW = np.full((CONCENTRATION.shape[0], 1), 0.2)
    X = np.full((CONCENTRATION.shape[0], 1), 505680.248)
    Y = np.full((CONCENTRATION.shape[0], 1), 2497248.282)
    TEST = np.hstack((X, Y, FLOW, CONCENTRATION))

    MDF_NAME = 'river'
    R_TIME = "2020-03-31 00:00:00"
    START, END = "2020-03-31 00:00:00", "2020-04-03 00:00:00"
    NETWORK = "4.7_model"
    RUN_TEMPLATE = "3.31-4.03"

    # 生成ICM观测值
    icm = InfoWorks(population=TEST, start=START, end=END, network=NETWORK,
                    run_template=RUN_TEMPLATE, plot=False)
    valid_random_num, valid_location = icm.creat_obs()

    # 生成Delft3D观测值
    args = (MDF_NAME, R_TIME, START, END, valid_random_num, TEST, valid_location)
    func = FunctionWrapper(delft3d_obs, args)
    pool = Pool(processes=30)
    pool.map(func, valid_random_num)







from ICM import *
from Delft3D import *
from func import FunctionWrapper
from multiprocessing.pool import Pool


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
    DIS_NAME, SRC_NAME = "river", "river"

    # 生成ICM观测值
    icm = InfoWorks(population=TEST, start=START, end=END, network=NETWORK,
                    run_template=RUN_TEMPLATE, plot=False)
    valid_random_num, valid_location = icm.creat_obs()

    # 生成Delft3D观测值
    delft3d_test = Delft3D(reference_time=R_TIME, start=START, end=END,
                           mdf_name=MDF_NAME,  dis_name=DIS_NAME,
                           src_name=SRC_NAME)

    args = (TEST, valid_random_num, valid_location)
    func = FunctionWrapper(delft3d_test.create_obs, args)
    pool = Pool(processes=30)
    pool.map(func, valid_random_num)

import configparser
from multiprocessing.pool import Pool

from Delft3D import *
from ICM import *
from func import FunctionWrapper

if __name__ == "__main__":
    # 生成待测试条件
    CONCENTRATION = np.arange(100, 2100, 100).reshape(-1, 1)
    FLOW = np.full((CONCENTRATION.shape[0], 1), 0.2)
    X = np.full((CONCENTRATION.shape[0], 1), 505680.248)
    Y = np.full((CONCENTRATION.shape[0], 1), 2497248.282)
    TEST = np.hstack((X, Y, FLOW, CONCENTRATION))

    # 读取配置文件
    cf = configparser.ConfigParser()
    cf.read("config.ini")
    R_TIME = cf.get("General", "reference_time")
    START = cf.get("General", "start")
    END = cf.get("General", "end")
    RUN_TEMPLATE = cf.get("InfoWorks", "run_template")
    SOURCE_TYPE = cf.get("General", "source_type")
    MDF_NAME = cf.get("Delft3D", "mdf_name")
    DIS_NAME = cf.get("Delft3D", "dis_name")
    SRC_NAME = cf.get("Delft3D", "src_name")
    NETWORK = cf.get("InfoWorks", "network")
    OBS_FOLDER = cf.get("General", "obs_folder")

    # 生成ICM观测值
    icm = InfoWorks(population=TEST, start=START, end=END, network=NETWORK,
                    run_template=RUN_TEMPLATE, source_type='instant')
    valid_random_num, valid_location = icm.creat_obs()

    # 生成Delft3D观测值
    delft3d_test = Delft3D(reference_time=R_TIME, start=START, end=END,
                           mdf_name=MDF_NAME,  dis_name=DIS_NAME,
                           src_name=SRC_NAME)

    args = (TEST, valid_random_num, valid_location)
    func = FunctionWrapper(delft3d_test.create_obs, args)
    pool = Pool(processes=30)
    pool.map(func, valid_random_num)

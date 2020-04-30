import configparser

from Delft3D import *
from scipy.optimize import differential_evolution

if __name__ == '__main__':
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
    # 观测点集

    # 运行脚本生成河流边界的时间序列，第一次运行需要运行这个！！！！！！
    os.system("C:/Users/Carlisle/Anaconda3/python.exe Delft3D_时间序列.py {} {} {}".format(R_TIME, START, END))

    delft3d_test = Delft3D(reference_time=R_TIME, start=START, end=END, mdf_name=MDF_NAME,
                           dis_name=DIS_NAME, src_name=SRC_NAME, obs_folder=OBS_FOLDER)
    # 运行算法
    bounds = [(505465, 506557), (2496786, 2497709), (0, 1), (300, 2000), (0, 288)]
    result = differential_evolution(delft3d_test.solve, bounds=bounds, start=START, end=END,
                                    network=NETWORK, run_template=RUN_TEMPLATE, obs_folder=OBS_FOLDER,
                                    source_type=SOURCE_TYPE, updating='deferred', workers=30,
                                    tol=0.0001, disp=True)
    print(result)


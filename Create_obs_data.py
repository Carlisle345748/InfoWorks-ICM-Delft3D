from multiprocessing.pool import Pool

from Delft3D import *
from ICM import *
from func import FunctionWrapper

if __name__ == "__main__":
    # generate parameters
    CONCENTRATION = np.arange(100, 2100, 100).reshape(-1, 1)
    FLOW = np.full((CONCENTRATION.shape[0], 1), 0.2)
    X = np.full((CONCENTRATION.shape[0], 1), 505680.248)
    Y = np.full((CONCENTRATION.shape[0], 1), 2497248.282)
    TEST = np.hstack((X, Y, FLOW, CONCENTRATION))

    # generate observed data for ICM
    icm = InfoWorks(TEST, config='config.ini')
    valid_random_num, valid_location = icm.create_obs()

    # generate observed data for Delft3D
    delft3d = Delft3D(config='config.ini')

    args = (TEST, valid_random_num, valid_location)
    func = FunctionWrapper(delft3d.create_obs, args)
    pool = Pool(processes=30)
    pool.map(func, valid_random_num)

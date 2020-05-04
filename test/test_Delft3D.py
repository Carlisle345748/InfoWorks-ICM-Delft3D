from unittest import TestCase, main
from multiprocessing.pool import Pool
from Delft3D import *


class TestDelft3D(TestCase):
    def test_solve(self):
        shutil.copyfile("obs/MH52_0.5_1000.0/Link_ds_cond.csv", "icm_to_delft3d/Link_1_ds_cond.csv")
        shutil.copyfile("obs/MH52_0.5_1000.0/Link_ds_flow.csv", "icm_to_delft3d/Link_1_ds_flow.csv")
        shutil.copyfile("obs/MH52_0.5_1000.0/Link_ds_cond.csv", "icm_to_delft3d/Link_2_ds_cond.csv")
        shutil.copyfile("obs/MH52_0.5_1000.0/Link_ds_flow.csv", "icm_to_delft3d/Link_2_ds_flow.csv")
        # Delft3D模块测试
        valid_random_num = [1, 2]
        delft3d_test = Delft3D(config='config_test.ini')
        pool = Pool(processes=2)
        delft3d_energies = pool.map(delft3d_test.solve, valid_random_num)
        self.assertEqual(delft3d_energies, [-1, -1])


if __name__ == '__main__':
    main()

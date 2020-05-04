from unittest import TestCase, main
from ICM import *


class TestInfoWorks(TestCase):
    def test_solve_continue(self):
        # ICM模块测试
        population = np.array([[505680.248, 2497248.282, 0.5, 1000],
                               [505680.248, 2497248.282, 0.5, 1000]])
        icm_test = InfoWorks(population, config='config_test.ini')

        icm_energies, valid_random_num = icm_test.solve()
        self.assertEqual(icm_energies, [-1, -1])


if __name__ == '__main__':
    main()

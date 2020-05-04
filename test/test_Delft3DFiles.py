import numpy as np
from unittest import TestCase, main
from Delft3DFiles import *


class TestDelft3DTimeSeries(TestCase):

    def test_bct_export(self):
        with open("bct_test.bct", 'r') as f:
            bct_test = f.readlines()
        bct_data = Delft3DTimeSeries("bct_test.bct")
        self.assertListEqual(bct_test, bct_data.export())

    def test_bcc_export(self):
        with open("bcc_test.bcc", 'r') as f:
            bcc_test = f.readlines()
        bcc_data = Delft3DTimeSeries("bcc_test.bcc")
        self.assertListEqual(bcc_test, bcc_data.export())

    def test_dis_export(self):
        with open("dis_test.dis", 'r') as f:
            dis_test = f.readlines()
        dis_data = Delft3DTimeSeries("dis_test.dis")
        self.assertListEqual(dis_test, dis_data.export())

    def test_set_header(self):
        bct_data = Delft3DTimeSeries("bct_test.bct")
        bct_data.set_header(0, {'location': '(1,1)..(1,1)'})
        bct_data.set_header(0, {'reference-time': '20200120'})
        bct_data.set_header(0, {'parameter': {'time': 'relative time'}})
        bct_data.set_header(0, {'parameter': {'time': 'hour'}}, unit=True)
        self.assertEqual(bct_data.bct_data[0].header['location'].value, '(1,1)..(1,1)')
        self.assertEqual(bct_data.bct_data[0].header['reference-time'].value, '20200120')
        self.assertEqual(bct_data.bct_data[0].header['parameter']['time'].value, 'relative time')
        self.assertEqual(bct_data.bct_data[0].header['parameter']['time'].unit, 'hour')

        self.assertEqual(bct_data.bct_data[0].header['location'].export(), "'(1,1)..(1,1)        '")
        self.assertEqual(bct_data.bct_data[0].header['reference-time'].export(), "20200120")
        self.assertEqual(bct_data.bct_data[0].header['parameter']['time'].export(),
                         "'relative time       '                    unit '[hour]'")

    def test_set_time_series(self):

        reference_time = pd.to_datetime("2020-04-15")
        time_index = pd.to_timedelta(np.arange(10)*10, unit="min") + reference_time
        data1 = pd.Series(np.ones(10) * 100, index=time_index)
        data2 = pd.Series(np.ones(10) * 200, index=time_index)

        relative_time = pd.to_timedelta(np.arange(10)*10, unit="min")
        relative_time = [time.total_seconds() / 60 for time in relative_time]
        test_table = pd.DataFrame({'time': relative_time, '1': np.ones(10) * 100,
                                   '2': np.ones(10) * 200}, index=time_index)

        bct_data = Delft3DTimeSeries("bct_test.bct")
        bct_data.set_time_series(0, "2020-04-15", data1, data2)

        self.assertTrue(np.array((bct_data.bct_data[0].time_series.index == time_index)).all())
        self.assertTrue((bct_data.bct_data[0].time_series.values == test_table.values).all())
        self.assertEqual(bct_data.bct_data[0].header['records-in-table'].value, str(len(time_index)))
        self.assertEqual(bct_data.bct_data[0].header['reference-time'].value, reference_time.strftime("%Y%m%d"))


if __name__ == '__main__':
    main()

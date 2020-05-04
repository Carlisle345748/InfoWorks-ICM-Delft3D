import re
import os
import pandas as pd


class Delft3DTimeSeries(object):
    def __init__(self, filename):
        self.type = os.path.splitext(filename)[1][1:]
        self.filename = filename
        self.bct_data = self.load_bct()

    def load_bct(self):
        with open(self.filename) as f:
            bct_data = f.readlines()
        start_index, end_index = 0, None
        time_series = []
        in_table = False
        for index, line in enumerate(bct_data):
            if in_table and index == len(bct_data) - 1:
                time_series.append(TimeSeries(bct_data[start_index:]))
            elif 'table-name' in line:
                end_index = index
                if in_table:
                    time_series.append(TimeSeries(bct_data[start_index: end_index]))
                    start_index = index
                else:
                    in_table = True
        return time_series

    def set_header(self, num: int, data: dict, unit=False):
        self.bct_data[num].set_header(data, unit)

    def set_time_series(self, num, reference_time, data1, data2):
        self.bct_data[num].set_time_series(reference_time, data1, data2)

    def export(self):
        bct_data = []
        for time_series in self.bct_data:
            bct_data += time_series.export()
        return bct_data

    def to_file(self, file):
        bct_data = self.export()
        with open(file, 'w') as f:
            for line in bct_data:
                f.write(line)


class TimeSeries(object):
    def __init__(self, time_series):
        self.time_series = time_series.copy()
        self.header = self.load_header()
        self.time_series = self.load_time_series()

    def load_header(self):
        header_dict = {}
        parameter = {}
        records_in_table = None
        header_re = re.compile(r"^([^-][\w-]+)\s+('?[\w\d (),./:-]+'?)")
        unit_re = re.compile(r"([\s]+unit '\[[\w/]+\]')")
        for index, line in enumerate(self.time_series):
            matches = header_re.search(line)
            if matches:
                if matches[1] == 'parameter':
                    unit_match = unit_re.search(line)
                    key_name = matches[2].strip('\'')
                    key_name = key_name.strip(' ')
                    parameter[key_name] = Parameter(matches[2], unit_match[1])
                elif matches[1] == 'records-in-table':
                    records_in_table = Parameter(matches[2])
                else:
                    header_dict[matches[1]] = Parameter(matches[2])
            else:
                header_dict['parameter'] = parameter
                header_dict['records-in-table'] = records_in_table
                break
        return header_dict

    def load_time_series(self):
        is_header = True
        reference_time = pd.to_datetime(self.header['reference-time'].value)
        time, parm1, parm2, parm3 = [], [], [], []
        for line in self.time_series:
            if not is_header:
                data = [float(i) for i in line.split()]
                time.append(reference_time + pd.to_timedelta(data[0], unit="minutes"))
                parm1.append(data[0])
                parm2.append(data[1])
                parm3.append(data[2])
            if 'records-in-table' in line:
                is_header = False
        else:
            colname = list(self.header['parameter'].keys())
            time_series = pd.DataFrame(
                {colname[0]: parm1, colname[1]: parm2, colname[2]: parm3}, index=time)
        return time_series

    def set_header(self, data: dict, unit=False) -> None:
        header = self.header.copy()
        for key, new_parm in data.items():
            if key not in ['parameter', 'reference-time', 'records-in-table']:
                header[key].value = str(new_parm)
            elif key in ['reference-time', 'records-in-table']:
                header[key].value = str(new_parm)
                print("修改reference-time和records-in-table要同时修改时间序列")
            else:
                for key_, new_parm_ in new_parm.items():
                    if unit:
                        header[key][key_].unit = str(new_parm_)
                    else:
                        header[key][key_].value = str(new_parm_)
        self.header = header

    def set_time_series(self, reference_time, data1, data2):
        """
        修改时间序列
        :param reference_time: 参照时间
        :param data2: 左列数据
        :param data1: 右列数据
        """
        reference_time = pd.to_datetime(reference_time)
        time_series = pd.concat([data1, data2], axis=1)
        relative_time = time_series.index - reference_time
        relative_time = [time.total_seconds() / 60 for time in relative_time]  # 单位：minute
        relative_time = pd.Series(relative_time, index=time_series.index, name='time')
        time_series = pd.concat([relative_time, time_series], axis=1)

        self.time_series = time_series.copy()
        reference_time = reference_time.strftime("%Y%m%d")
        self.set_header({'records-in-table': len(time_series), "reference-time": reference_time})

    def export_header(self):
        header = []
        for key, parm in self.header.items():
            if key != 'parameter':
                head = key.ljust(21) + parm.export() + '\n'
                header.append(head)
            else:
                for i in parm:
                    head = key.ljust(21) + parm[i].export() + '\n'
                    header.append(head)
        return header

    def export_time_series(self):
        time_series = []
        for index, row in self.time_series.iterrows():
            time_series.append(" {:.7e} {:.7e} {:.7e}\n".format(row[0], row[1], row[2]))
            pass
        return time_series

    def export(self):
        return self.export_header() + self.export_time_series()


class Parameter(object):
    def __init__(self, value, unit=None):
        value_re = re.compile(r'[\w() /:,.-]+\b\)?')
        value_match = value_re.search(value)
        self.value = value_match[0]
        if '\'' in value:
            self.value_length = len(value) - 2
            self.type = 'str'
        else:
            self.value_length = len(value)
            self.type = 'num'

        self.unit = None
        if unit:
            unit_re = re.compile(r"unit '\[([\w/]+)\]'")
            unit_match = unit_re.search(unit)
            self.unit = unit_match[1]
            self.unit_length = len(unit)

    def export(self):
        if self.type == 'str':
            content = "'{}'".format(self.value.ljust(self.value_length))
            if self.unit:
                content += ("unit '[{}]'".format(self.unit)).rjust(self.unit_length)
        else:
            content = "{}".format(self.value.ljust(self.value_length))

        return content

    def __str__(self):
        if self.unit:
            return "{} unit={}".format(self.value, self.unit)
        else:
            return "{}".format(self.value)
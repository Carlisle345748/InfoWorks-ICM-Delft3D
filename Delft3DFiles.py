import re
import os
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
import matplotlib.pyplot as plt


class TimeSeriesFile(object):
    """
    Read, modify, export and write Delft3D time series files (bcc/bct/dis).

    Examples
    --------
    >>> bct = TimeSeriesFile('river.bct')
    >>> bcc = TimeSeriesFile('river.bcc')
    >>> dis = TimeSeriesFile('river.dis')
    """
    def __init__(self, filename):
        self.type = os.path.splitext(filename)[1][1:]
        self.filename = filename
        self.data = self.load_file()

    def load_file(self):
        """Read bct/bcc/dis file. The content of the file will be stored in self.data. """
        with open(self.filename) as f:
            data = f.readlines()
        # read each time series and interpret them
        time_series = []
        start_index, end_index = 0, None
        in_table = False
        for index, line in enumerate(data):
            if in_table and index == len(data) - 1:
                # last time series
                time_series.append(TimeSeries(data[start_index:]))
            elif 'table-name' in line:
                end_index = index
                if in_table:
                    # interpret time series
                    time_series.append(TimeSeries(data[start_index: end_index]))
                    start_index = index
                else:
                    in_table = True
        return time_series

    def set_header(self, num, data, unit=False):
        """
        Modify the content of the header. IMPORTANT: This method cannot change the value and
        the unit of a parameter simultaneously. If you want to change the unit of a parameter,
        please set unit=True and only change the unit in one call.

        Parameters
        ----------
        num : int
            the no. of time series. '0' means the first time series in the file.
        data : dict
            a dict contains new content of headers, e.g. {'reference-time': '20200304'}
        unit : bool, optional
            If true, this method change the unit of parameter. Otherwise thsi method
            change the value of parameter.

        Returns
        -------

        Examples
        -------
        >>> bct = TimeSeriesFile('river.bct')
        >>> bct.set_header(0, {'time-unit': 'hours', 'location': '(2,3)..(4,6)'})
        >>> bct.set_header(0, {'parameter': {'time': 'relative-time', 'pollution': 'NH3-N'}})
        >>> bct.set_header(0, {'parameter':{'time': 'hour', 'pollution': 'mg/l'}}, unit=True)

        """
        self.data[num].set_header(data, unit)

    def set_time_series(self, num, reference_time, data1, data2):
        """
        Replace the old time series with the new one.

            Parameters
            ----------
            num : int
                The no. of time series. '0' means the first time series in the file.
            reference_time: str
                new reference time
            data1: pd.DataFrame or pd.Series
                The second column of time series.
                The index of the Series must be DatetimeIndex
            data2 : pd.DataFrame or pd.Series
                The thrid column of time series.
                The index of the Series must be DatetimeIndex


        Example
        ----------
        >>> bct = TimeSeriesFile('river.bct')
        >>> flow_series = pd.read_csv('flow.csv', index_col=0)
        >>> pollution_series = pd.read_csv('pollution.csv', index_col=0)
        >>> flow_series.head()
            2020-03-31 00:00:00    0.4077
            2020-03-31 00:10:00    0.4282
            2020-03-31 00:20:00    0.4707
            2020-03-31 00:30:00    0.5127
            2020-03-31 00:40:00    0.5692
        >>> bct.set_time_series(0, '2020-03-04', flow_series, pollution_series)

        """
        self.data[num].set_time_series(reference_time, data1, data2)

    def export(self):
        """
        Export the data to a list in the format of Delft3D time series file.

        Example
        -------
        >>> bct = TimeSeriesFile('river.bct')
        >>> bct_file = bct.export()
        >>> bct_file
            ["table-name           'Boundary Section : 1'\\n",
             "contents             'Uniform             '\\n",
             "location             '(2,246)..(7,246)    '\\n",
            ...]

        """
        bct_data = []
        for time_series in self.data:
            bct_data += time_series.export()
        return bct_data

    def to_file(self, filename):
        """
        Write the data to a Delft3D time series file.

        Parameters
        ----------
        filename : str
            Filename of the time series file

        Examples
        ----------
        >>> bct = TimeSeriesFile('river.bct')
        >>> bct.to_file('river.bct')
        """
        bct_data = self.export()
        with open(filename, 'w') as f:
            for line in bct_data:
                f.write(line)


class TimeSeries(object):
    """Read, modify and export Delft3D time series."""
    def __init__(self, time_series: list):
        self.time_series = None
        self.header = None
        self.load_header(time_series)
        self.load_time_series(time_series)

    def load_header(self, time_series: list):
        """Read and interpret the header of a time series."""
        header_dict = {}
        parameter = {}
        records_in_table = None
        header_re = re.compile(r"^([^-][\w-]+)\s+('?[\w\d (),./:-]+'?)")
        unit_re = re.compile(r"([\s]+unit '\[[\w/]+\]')")
        for line in time_series:
            matches = header_re.search(line)  # search for header
            if matches:
                if matches[1] == 'parameter':
                    # parameters have the same header name. So store all parameters
                    # in one dict
                    unit_match = unit_re.search(line)  # search for unit
                    key_name = matches[2].strip('\'')  # reformat unit
                    key_name = key_name.strip(' ')
                    parameter[key_name] = Parameter(matches[2], unit_match[1])
                elif matches[1] == 'records-in-table':
                    # records-in-table should be the last header. Store it hera and
                    # then put it at the end of headers by the end.
                    records_in_table = Parameter(matches[2])
                else:
                    # regular header
                    header_dict[matches[1]] = Parameter(matches[2])
            else:  # end of the header
                header_dict['parameter'] = parameter
                header_dict['records-in-table'] = records_in_table
                break
        self.header = header_dict

    def load_time_series(self, time_series: list):
        """Read and interpret time series"""
        is_header = True  # whether the pointer at the header
        reference_time = pd.to_datetime(self.header['reference-time'].value)
        # read the time series data
        time, relative_time, parm1, parm2 = [], [], [], []
        for line in time_series:
            if not is_header:
                # prepossess
                data = [float(i) for i in line.split()]
                time.append(reference_time + pd.to_timedelta(data[0], unit="minutes"))
                # store the data
                relative_time.append(data[0])
                parm1.append(data[1])
                parm2.append(data[2])
            if 'records-in-table' in line:
                is_header = False
        else:
            # converts lists to DataFrame
            colname = list(self.header['parameter'].keys())
            time_series = pd.DataFrame(
                {colname[0]: relative_time, colname[1]: parm1, colname[2]: parm2}, index=time)
        self.time_series = time_series

    def set_header(self, data: dict, unit=False) -> None:
        """Set new content of header. Called by TimeSeriesFile.set_header()"""
        header = self.header.copy()
        for key, new_parm in data.items():
            if key not in ['parameter', 'reference-time', 'records-in-table']:
                # regular header
                header[key].value = str(new_parm)
            elif key in ['reference-time', 'records-in-table']:
                # raise warning when reference-time and records-in-table are changed
                header[key].value = str(new_parm)
                print("'reference-time' and 'records-in-table' have been changed."
                      " Please check time series data")
            else:
                # change parameter
                for key_, new_parm_ in new_parm.items():
                    if unit:
                        header[key][key_].unit = str(new_parm_)
                    else:
                        header[key][key_].value = str(new_parm_)
        self.header = header

    def set_time_series(self, reference_time: str,
                        data1: pd.core.frame.Series,
                        data2: pd.core.frame.Series):
        """
        Replace the old time series with the new one. Called by TimeSeriesFile.set_time_series()
        """
        time_series = pd.concat([data1, data2], axis=1)
        # calculate the absolute time and  relative time
        reference_time = pd.to_datetime(reference_time)
        relative_time = time_series.index - reference_time
        relative_time = [time.total_seconds() / 60 for time in relative_time]  # 单位：minute
        relative_time = pd.Series(relative_time, index=time_series.index, name='time')
        # combine time absolute time, relative time and data
        time_series = pd.concat([relative_time, time_series], axis=1)
        # store new time series
        self.time_series = time_series.copy()
        # change the 'reference time' and 'records-in-table' in the header
        reference_time = reference_time.strftime("%Y%m%d")
        self.set_header({'records-in-table': len(time_series), "reference-time": reference_time})

    def export_header(self):
        """Export the header as a list in the format of Delft3D time series file"""
        header = []
        for key, parm in self.header.items():

            if key != 'parameter':
                # parameter header
                head = key.ljust(21) + parm.export() + '\n'
                header.append(head)
            else:
                # regular header
                for i in parm:
                    head = key.ljust(21) + parm[i].export() + '\n'
                    header.append(head)
        return header

    def export_time_series(self):
        """Export the time series as a list in the format of Delft3D time series files"""
        time_series = []
        for index, row in self.time_series.iterrows():
            time_series.append(" {:.7e} {:.7e} {:.7e}\n".format(row[0], row[1], row[2]))
            pass
        return time_series

    def export(self):
        """Export all data as a list in the format of Delft3D time series files"""
        return self.export_header() + self.export_time_series()


class Parameter(object):
    """
    Read and export the content of header in Delft3D Time Series. The function of this class
    is to keep the original format of Delft3D Time Series in order to prevent unexpected errors.
    """
    def __init__(self, value, unit=None):
        """Read the store the format, type and unit of a header"""
        value_re = re.compile(r'[\w() /:,.-]+\b\)?')  # search for the value
        value_match = value_re.search(value)
        self.value = value_match[0]
        if '\'' in value:
            # string type
            self.value_length = len(value) - 2  # length of the string
            self.type = 'str'
        else:
            # number type
            self.value_length = len(value)  # length of the number
            self.type = 'num'

        self.unit = None
        # store the unit
        if unit:
            # search for unit
            unit_re = re.compile(r"unit '\[([\w/]+)\]'")
            unit_match = unit_re.search(unit)
            # store the unit
            self.unit = unit_match[1]
            self.unit_length = len(unit)

    def export(self):
        """export the header in its original format"""
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


class MdfFile(object):
    """
    Read, modify, export and write the Delft3D mdf file

    Examples
    --------
    >>> mdf = MdfFile('river.mdf')
    """
    def __init__(self, filename):
        self.filename = filename
        self.data = self.load_file()

    def load_file(self):
        """Read mdf file and store it in self.data as a dict"""
        with open(self.filename, 'r') as f:
            mdf_data = f.readlines()
        # read MDF File and store it in a dict
        mdf_dict = {}
        parm_name = None  # name of the parameter
        for line in mdf_data:
            # search for the name and value of parameters
            matches = re.search(r'^([\w]+)\s*=\s*([\w .#+-:\[\]]*)$', line)
            if matches is not None and matches[1] != 'Commnt':
                # single-line parameter
                # ignore the Comment which are unused by the model
                parm_name = matches[1]  # name of the parameter
                if '#' in matches[2] and '[' not in matches[2]:
                    # character parameter 1
                    mdf_dict[matches[1]] = matches[2].rstrip(' ')
                    mdf_dict[matches[1]] = mdf_dict[matches[1]].replace('#', '')
                elif '[' in matches[2]:
                    # character parameter 2
                    mdf_dict[matches[1]] = matches[2]
                else:
                    # single number parameter
                    num = [float(x) for x in matches[2].split()]
                    # array parameter
                    mdf_dict[matches[1]] = np.array(num) if len(num) > 1 else num[0]
            elif matches is None:
                # multiple-line parameter
                matches = re.search(r'^\s+(.*)$', line)  # search for the value
                parm = mdf_dict.get(parm_name)  # find the name in the last recorded parameter
                if '#' in line:
                    # multiple-line character parameter
                    parm = parm if type(parm) == list else [parm]
                    parm.append(matches[1].replace('#', ''))
                else:
                    # multiple-line array parameter
                    parm = np.array(parm).reshape(-1, 1)
                    parm = np.append(parm, np.array(float(matches[1])).reshape(1, 1), axis=0)
                # store multiple-line parameter
                mdf_dict[parm_name] = parm

        return mdf_dict

    def set_parm(self, data):
        """
        Set new value for a parameter. When setting new values for parameters with multiple
        values (single-line or multiple-line array parameter e.g. Flmap), please input iterable
        data type such as list, tuple and ndarray.

        Parameters
        ----------
        data : dict
            A dict contains names and new values of parameters e.g. {'Fildep': 'river.dep'}.
            Each key and value are corespond to the name and value of one parameter.

        Examples
        ----------
        >>> mdf = MdfFile('river.mdf')
        >>> mdf.set_parm({'Fildep': 'river.dep', 'Dt': 0.5, 'Flmap':[0, 10, 4320]})

        """
        for key, value in data.items():
            if type(self.data[key]) in [float, int]:
                # single number parameter
                self.data[key] = float(value)
            elif type(self.data[key]) == np.ndarray:
                # array parameter
                if len(self.data[key].shape) == 1:
                    # single-line array parameter
                    self.data[key] = np.array(value)
                else:
                    # multiple-line array parameter
                    self.data[key] = np.array(value).reshape(-1, 1)
            elif key == 'Runtxt':
                # multiple-line character parameter
                self.data[key] = value
            else:
                # character parameter
                self.data[key] = str(value)

    def export(self):
        """
        Export the data to a list in the format of Delft3D mdf file

        Examples:
        ---------
        >>> mdf = MdfFile('river.mdf')
        >>> mdf_file = mdf.export()
        >>> mdf_file
            ['Ident  = #Delft3D-FLOW 3.59.01.57433#\\n',
             'Filcco = #river.grd#\\n',
             'Anglat = 2.2560000e+01\\n',
             'Grdang = 2.2830000e+02\\n',
             ...]
        """
        mdf_file = []
        int_key = ['MNKmax', 'Ktemp', 'Ivapop', 'Irov', 'Iter']  # integer parameters
        for key, content in self.data.items():
            if type(content) == np.ndarray and len(content.shape) > 1:
                # multiple-line array parameter
                formatter = "%-6s = %d\n" if key in int_key else "%-6s = %.7e\n"
                mdf_file.append(formatter % (key, content[0]))  # first line
                for arr in content[1:]:  # the rest lines
                    formatter = "          %d\n" if key in int_key else "          %.7e\n"
                    mdf_file.append(formatter % arr)

            elif type(content) == np.ndarray and len(content.shape) == 1:
                # array parameter
                line = "%-6s =" % key
                for arr in content:
                    line += " %d" % arr if key in int_key else " %.7e" % arr
                mdf_file.append(line + '\n')

            elif type(content) == list:
                # multiple-line character parameter
                mdf_file.append("%-6s = #%s#\n" % (key, content[0]))
                for line in content[1:]:
                    mdf_file.append("         #%s#\n" % line)

            else:
                if type(content) == float:
                    # single number parameter
                    formatter = "%-6s = %d" if key in int_key else "%-6s = %.7e"
                    line = formatter % (key, content)
                elif type(content) == str and '[' not in content:
                    # single character parameter 1
                    line = "%-6s = #%s#" % (key, content)
                elif type(content) == str and '[' in content:
                    # single character parameter 2
                    line = "%-6s = %s" % (key, content)
                else:
                    raise ValueError("invalid key")
                mdf_file.append(line + '\n')

        return mdf_file

    def to_file(self, filename):
        """
        Write the data to a Delft3D mdf file

        Parameters
        ----------
        filename : str
            Filename of the mdf file

        Examples
        --------
        >>> mdf = MdfFile('river.mdf')
        >>> mdf.to_file('river.mdf')
        """
        mdf_file = self.export()
        with open(filename, 'w') as f:
            f.writelines(mdf_file)


class GrdFile(object):
    """
    Read, modify, visualize, export and write Deflt3D dep file

    Examples
    --------
    >>> grd = GrdFile('river.grd')
    """
    def __init__(self, filename: str):
        self.filename = filename
        self.x, self.y = None, None
        self.header = {}
        self.load_file()

    def load_file(self):
        """Read dep file"""
        with open(self.filename, 'r') as f:
            data = f.read()
        # read headers
        coordinate_system = re.search(r'Coordinate System = ([\w]+)', data)
        self.header['Coordinate System'] = coordinate_system.group(1) if coordinate_system else None
        missing_value = re.search(r'Missing Value\s+=\s+([\w+-.]+)', data)
        self.header['Missing Value'] = np.float(missing_value.group(1)) if missing_value else None
        mn = re.search(r'\n\s+([\d]+)\s+([\d]+)\n', data)
        m, n = int(mn.group(1)), int(mn.group(2))
        self.header['MN'] = [m, n]
        # read coordinates
        x, y = [], []
        pattern = r' ETA=\s+\d+(\s+[\d.Ee+]+\n?){' + str(m) + '}'
        matches = re.finditer(pattern, data)
        for index, match in enumerate(matches):
            cor = match[0].split()[2:]
            cor = [np.float(num) for num in cor]
            if index < n:
                x.extend(cor)
            else:
                y.extend(cor)
        x, y = np.array(x), np.array(y)
        # mask invalid value
        x = np.ma.masked_equal(x, self.header['Missing Value']) if missing_value else x
        y = np.ma.masked_equal(y, self.header['Missing Value']) if missing_value else y
        # reshape to the original format
        self.x = x.reshape(n, m)
        self.y = y.reshape(n, m)

    def spherical_to_cartesian(self, sph_epsg=4326, car_epsg=3857):
        """
        Convert from spherical coordinates to cartesian coordinates.
        Default spherical coordinate system: WGS 84.
        Default cartesian coordinate system: WGS_1984_Web_Mercator_Auxiliary_Sphere.
        Find the EPSG of more coordinate system in the following link.
        https://developers.arcgis.com/javascript/3/jshelp/pcs.htm

        Parameters
        ----------
        sph_epsg : int, optional
            EPSG of the original spherical coordinate system
        car_epsg : int, optional
            EPSG of the objective cartesian coordinate system

        Examples
        ----------
        >>> grd = GrdFile('river.grd')
        >>> grd.spherical_to_cartesian()
        >>> grd.spherical_to_cartesian(sph_epsg=4326, car_epsg=26917)
        """
        # transform from spherical to cartesian
        init_crs = CRS.from_epsg(sph_epsg)
        obj_crs = CRS.from_epsg(car_epsg)
        projection = Transformer.from_crs(init_crs, obj_crs)
        # update x, y
        self.x, self.y = projection.transform(self.x, self.y)
        # update header
        self.header['Coordinate System'] = 'Cartesian'

    def cartesian_to_spherical(self, car_epsg=3857, sph_epsg=4326):
        """
        Convert from cartesian coordinates to spherical coordinates.
        Default spherical coordinate system: WGS 84.
        Default cartesian coordinate system: WGS_1984_Web_Mercator_Auxiliary_Sphere.
        Find the EPSG of more coordinate system in the following link.
        https://developers.arcgis.com/javascript/3/jshelp/pcs.htm

        Parameters
        ----------
        car_epsg : int, optional
            EPSG of the original cartesian coordinate system
        sph_epsg : int, optional
            EPSG of the objective spherical coordinate system
        Examples
        ----------
        >>> grd = GrdFile('river.grd')
        >>> grd.cartesian_to_spherical()
        >>> grd.cartesian_to_spherical(car_epsg=26917, sph_epsg=4326)

        """
        # transform from cartesian to spherical
        init_crs = CRS.from_epsg(car_epsg)
        obj_crs = CRS.from_epsg(sph_epsg)
        projection = Transformer.from_crs(init_crs, obj_crs)
        # update x, y
        self.x, self.y = projection.transform(self.x, self.y)
        # update header
        self.header['Coordinate System'] = 'Spherical'

    def get_nearest_grid(self, x, y, sph_epsg=4326, car_epsg=3857):
        """
        Find the nearest grid for the giving coordinate. If the coordinate system is
        spherical, it will be automatically convert to cartesian coordinate system.
        You can specify the EPSG of coordiante by assigning sph_egsp and car_epsg.
        Find the EPSG of more coordinate system in the following link.
        https://developers.arcgis.com/javascript/3/jshelp/pcs.htm

        Parameters
        ----------
        x : float
            x coordinate.
        y : float
            y coordinate.
        sph_epsg : int, optional
            The EPSG of spherical cooridante.
        car_epsg : int, optional
            The EPSG of carsetian cooridante.
        Returns
        -------
        m, n : tuple
            (m,n) coordinate of grid

        Examples
        --------
        >>> grd = GrdFile('river.grd')
        >>> m, n = grd.get_nearest_grid(505944.89, 2497013.47)
        """
        if self.header['Coordinate System'] == 'Spherical':
            # transform from spherical to cartesian
            grd_crs = CRS.from_epsg(sph_epsg)
            plot_crs = CRS.from_epsg(car_epsg)
            projection = Transformer.from_crs(grd_crs, plot_crs)
            grd_x, grd_y = projection.transform(self.x, self.y)
            print("Automatically transform from spherical to cartesian coordinates.\n"
                  "Change the default projection by giving specific grd_epsg and plot_epsg")
        else:
            grd_x, grd_y = self.x, self.y
        # calculate distance
        dis = np.sqrt(
            (x - grd_x.ravel()) ** 2 + (y - grd_y.ravel()) ** 2)
        # find nearest grid
        num = np.argmin(dis)
        n, m = np.unravel_index(num, (self.header['MN'][1], self.header['MN'][0]))
        return m, n

    def plot(self, sph_epsg=4326, car_epsg=3857):
        """
        Visualize the grid.If the coordinate system is spherical, it will be automatically
        convert to cartesian coordinate system. You can specify the EPSG of coordiante
        by assigning sph_egsp and car_epsg. Find the EPSG of more coordinate system in
        the following link. https://developers.arcgis.com/javascript/3/jshelp/pcs.htm
        Parameters
        ----------
        sph_epsg : int, optional
        car_epsg : int, optional

        Examples
        -------
        >>> grd = GrdFile('river.grd')
        >>> grd.plot()
        >>> grd.plot(sph_epsg=4326, car_epsg=26917)
        """
        if self.header['Coordinate System'] == 'Spherical':
            # transform from spherical to cartesian
            grd_crs = CRS.from_epsg(sph_epsg)
            plot_crs = CRS.from_epsg(car_epsg)
            projection = Transformer.from_crs(grd_crs, plot_crs)
            x, y = projection.transform(self.x, self.y)
            print("Automatically transform from spherical to cartesian coordinates.\n"
                  "Change the default projection by giving specific grd_epsg and plot_epsg")
        else:
            x, y = self.x, self.y
        # plot grid
        plt.pcolormesh(x, y, np.zeros(np.shape(self.x)),
                       edgecolor=None, facecolor='none', linewidth=0.005)
        plt.axis('equal')
        plt.show()

    def set_gird(self, x, y, coordinate_system):
        """
        Set new grid.

        Parameters
        ----------
        x : ndarray
            x coordinates of the new grid
        y : ndarray
            y coordinates of the new grid
        coordinate_system : str
            The type of coordinate system. Spherical or Cartesian

        Examples
        -------
        >>> grd = GrdFile('river.grd')
        >>> grd_x = np.loadtxt('grd_x.txt')
        >>> grd_y = np.loadtxt('grd_y.txt')
        >>> grd.set_gird(grd_x, grd_y, 'Cartesian')
        """
        self.x = x
        self.y = y
        self.header['Coordinate System'] = coordinate_system
        self.header['MN'] = [x.shape[1], x.shape[0]]

    def export(self):
        """
        Export the data to a list in the format of Delft3D grd file.

        Examples
        -------
        >>> grd = GrdFile('river.grd')
        >>> grd_file = grd.export()
        >>> grd_file
            ['Coordinate System = Cartesian\\n',
             'Missing Value = -9.9999900e+02\\n',
             '       7     245\\n',
             ' 0 0 0\\n',
             ...]
        """
        grd_file = list()
        # Add header
        grd_file.append("Coordinate System = %s\n" % self.header['Coordinate System'])
        if self.header['Missing Value'] is not None:
            grd_file.append("Missing Value = %.7e\n" % self.header['Missing Value'])
        grd_file.append("%8d%8d\n" % ((self.header['MN'][0]), self.header['MN'][1]))
        grd_file.append(" 0 0 0\n")
        # Add grid data
        grd_file = self.grid_writer(grd_file, self.x)
        grd_file = self.grid_writer(grd_file, self.y)

        return grd_file

    @staticmethod
    def grid_writer(grd_file, coordinates):
        """Helper function of self.export. Formatting grid data as Delft3D grd file"""
        grd_file = grd_file.copy()
        for index, cor in enumerate(coordinates):
            line = " ETA=%5d" % (index + 1)
            counts = 0
            for num in cor:
                if counts == 0:
                    line += "   %.17E" % num
                elif counts % 5 == 4:
                    line += "   %.17E\n" % num
                elif counts % 5 == 0:
                    line += "             %.17E" % num
                else:
                    line += "   %.17E" % num
                if counts == len(cor) - 1 and counts % 5 != 4:
                    line += '\n'
                counts += 1
            grd_file.append(line)
        return grd_file

    def to_file(self, filename):
        """
        Write the data to a Delft3D grd file.

        Parameters
        ----------
        filename : str
            Filename of the grd file.

        Examples
        -------
        >>> grd = GrdFile('river.grd')
        >>> grd.to_file('river.grd')
        """
        grd_file = self.export()
        with open(filename, 'w') as f:
            f.writelines(grd_file)


class DepFile(object):
    """
    Read, modify, visualize, export and write Delft3D dep file

    Example
    --------
    >>> dep = DepFile('river.dep')
    """
    def __init__(self, filename):
        self.filename = filename
        self.data = self.load_dep()

    def load_dep(self):
        """Read dep file"""
        with open(self.filename, 'r') as f:
            data = f.readlines()
        dep = list()
        for line in data:
            dep.append([float(i) for i in line.split()])
        dep = np.array(dep)
        dep = np.delete(dep, -1, axis=0)
        dep = np.delete(dep, -1, axis=1)

        return dep

    def plot(self, grd_file):
        """
        Visualize dep file

        Parameters
        ----------
        grd_file : GrdFile
            GrdFile instance of the corresponding grd file.

        Examples
        -------
        >>> grd = GrdFile('river.grd')
        >>> dep = DepFile('river.dep')
        >>> dep.plot(grd)
        """
        if type(grd_file) != GrdFile:
            raise ValueError("Please input an GrdFile class instance")
        if grd_file.header['Coordinate System'] == 'Spherical':
            grd_file.spherical_to_cartesian()
            print("Automatically transform from spherical to cartesian coordinates")
        plt.pcolormesh(grd_file.x, grd_file.y, self.data, cmap='Blues',
                       edgecolor=None, linewidth=0.005)
        plt.axis('equal')
        plt.colorbar()
        plt.show()

    def set_dep(self, data):
        """
        set new dep data
        Parameters
        ----------
        data : ndarray
            New dep data.
        Examples
        -------
        >>> dep = DepFile('river.deo')
        >>> dep_data = np.loadtxt('dep_data.txt')
        >>> dep.set_dep(dep_data)
        """
        self.data = data

    def export(self):
        """
        Export the data to a ndarry in the format of Delft3D dep file.

        Examples
        -------
        >>> dep = DepFile('river.dep')
        >>> dep_file = dep.export()
        >>> dep_file
            ['   1.6929708E-01   2.8992051E-01   5.0572435E-01\\n,
             '  -5.0850775E-02   3.1147481E-01   4.6392793E-01\\n,
             ...]
        """
        dep_data = np.append(self.data, np.full((1, self.data.shape[1]), -999.0), axis=0)
        dep_data = np.append(dep_data, np.full((dep_data.shape[0], 1), -999.0), axis=1)
        dep_file = []
        for line in list(dep_data):
            temp = []
            for num in line:
                temp.append("%16.7E" % num)
            temp = ''.join(temp) + '\n'
            dep_file.append(temp)
        return dep_file

    def to_file(self, filename):
        """
        Write the data to a Delft3D dep file.

        Parameters
        ----------
        filename : str
            Filename of Delft3D dep file
        Examples
        -------
        >>> dep = DepFile('river.dep')
        >>> dep.to_file('river.dep')
        """
        dep_file = self.export()
        with open(filename, 'w') as f:
            f.writelines(dep_file)

"""
write something here!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Circle
import h5py
import os
import json
import datetime
import seaborn as sns
import warnings
from typing import Dict

# some global variables
notebook_folder = os.getcwd()
main_folder = os.path.dirname(notebook_folder)


def get_salar_shape():
    """
    reference polygon to delimiter the salar in plots.
    """

    salar = np.array([[-68.25807007947564, -20.41030073970572], [-68.17211870291798, -20.55666113265817],
                      [-68.02037901826139, -20.50581596853763], [-68.00989672976979, -20.38071814594258],
                      [-67.95745041577791, -20.24886221185597], [-67.88349805080669, -20.30915586658684],
                      [-67.86889309558192, -20.38323208980657], [-67.86433483629197, -20.5013320625563],
                      [-67.80248429216705, -20.49443294173213], [-67.67138865247993, -20.43198447245249],
                      [-67.57522980534151, -20.58017764350799], [-67.3279086084335, -20.70761311742354],
                      [-67.12289413344833, -20.60527116811986], [-66.9308086175656, -20.40113955672989],
                      [-66.95345163530368, -20.32248330685868], [-67.15139305115768, -20.15180825456438],
                      [-67.18885191010472, -19.89753041675914], [-67.34395016539897, -19.71722570358426],
                      [-67.55771921908784, -19.71041323063233], [-67.56261826791018, -19.88399484408756],
                      [-67.72944737954316, -19.90267938006858], [-68.02550227004737, -19.78560798986577],
                      [-68.19385010814534, -19.937802407731], [-68.23701647699815, -20.27121569796243],
                      [-68.25807007947564, -20.41030073970572] ])
    return np.transpose(salar)


def read_clima_data(plot_stats: bool = True,
                    print_stats: bool = False,
                    export_plot: bool = True) -> pd:
    """
    Function to read the weather data from a xls file
    :param plot_stats: true if you want to plot the data
    :param print_stats: true if you want to print the file
    :param export_plot: true if you want to save the plots in the output folder
    :return: pandas structure with the clima
    """
    # read data
    route_to_file = r'../satellite_data/dato_diarios_uyuni_aeropuerto.csv'
    clima = pd.read_csv(route_to_file, encoding='unicode_escape', engine='python')
    print('----> Reading climate data from file: ', route_to_file)
    if print_stats is True:
        print('CLIMA = \n', clima)

    # transform dates
    cols = ['gestion', 'mes', 'dia']
    clima['date'] = clima[cols].apply(lambda x: '-'.join(x.values.astype(str)), axis="columns")
    clima['date'] = pd.to_datetime(clima['date'])

    # replace empty values by nan
    clima['Precipitación'] = clima['Precipitación'].replace('', 0)
    clima = clima.mask(clima == '')

    if print_stats is True:
        print(clima)
        print(type(clima))
        print(clima.shape)

    if plot_stats is True:
        fig, ax = plt.subplots()
        ax.set_xlabel('Date')
        ax.set_ylabel('T(°)')
        ax.plot(clima['date'], clima['Temperatura Máxima'], linewidth=2.0, color='r', label='T° max')
        ax.plot(clima['date'], clima['Temperatura Mínima'], linewidth=2.0, color='b', label='T° min')
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

        # vlines
        month = ['Jan', 'Feb', 'Mar', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan']
        date = ['20160101', '20160201', '20160301', '20160401', '20160501', '20160601',
                '20160701', '20160801', '20160901', '20161001', '20161101', '20161201', '20170101']
        date1 = []
        for el in date:
            date1.append(datetime.datetime.strptime(el, '%Y%m%d').date())

        for i in range(0, len(month)):
            ax.axvline(date1[i], linewidth=0.5, linestyle='--', color='cyan')
            ax.text(date1[i], max(clima['Temperatura Máxima']) + 4,
                    month[i], rotation=90, verticalalignment='center', color='cyan')

        ax2 = ax.twinx()
        ax2.set_ylabel('Rain (mm)')
        ax2.bar(clima['date'], clima['Precipitación'], color='g', label='rain', linewidth=2.0)

        ax.set_title('Weather Uyuni Airport, Year 2016')

        max_val = max(clima['Precipitación']) \
            if max(clima['Precipitación']) > max(clima['Temperatura Máxima']) else max(clima['Temperatura Máxima'])
        min_val = min(clima['Temperatura Mínima'])
        ax.set_ylim(min_val-4, max_val+4)
        ax2.set_ylim(min_val-4, max_val+4)

        fig.legend(loc='upper right')
        plt.grid()

        if export_plot is True:
            plt.savefig('../plots/clima_plot.pdf')
            plt.savefig('../plots/clima_plot.png')

            if print_stats:
                print('Clima plots save in {}'.format(main_folder + '/plots'))

        plt.show()

    print('----> Climate data was read correctly\n')

    return clima


def average_of_array(vector: list):
    """
    Compute the average of the observations
    :param vector: list with the data
    :return:
    """
    if type(vector) is not list:
        raise ValueError('Input vector must be an array')
    if len(vector) < 1:
        raise ValueError('Length of the vector must be at least 1')
    return sum(vector)/len(vector)


def read_h5_smap_l2(file_route: str,  # folder where the files are located
                    station_coordinates: np.ndarray,
                    inner_polygon: np.ndarray,
                    outer_polygon: np.ndarray,
                    plot_stats: bool = False,
                    plot_data: bool = False):
    """
    Read *.h5 data from the satellite mission which was operational some months in 2015:
    - SMAP L2 Radar/Radiometer Half-Orbit 9 km EASE-Grid Soil Moisture, Version 3 (SMAP_L2_SM_AP_01080_D)
      see: https://nsidc.org/data/SPL2SMAP/versions/3

    Read an h5 image and export the points inside a polygon
    Create a polygon in: http://geojson.io/
    """

    # 1 create environment

    # get files inside the folder
    files = os.listdir(file_route + '\\')
    print('----> Reading satellite data from folder {}'.format(file_route))

    # create folder where output is going to be stored
    out_path = os.path.join(file_route + '_outputs\\')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print('Output folder was created in {}'.format(out_path))
    else:
        print('Output folder already exists {}'.format(out_path))

    # define workplace
    # define polygons
    poly_lon, poly_lat, in_poly_lon, in_poly_lat = [], [], [], []
    for i in range(0, len(outer_polygon)):
        poly_lon.append(outer_polygon[i][0])
        poly_lat.append(outer_polygon[i][1])
        in_poly_lon.append(inner_polygon[i][0])
        in_poly_lat.append(inner_polygon[i][1])

    dic_data = {}
    cnt = 1

    # 2 open files
    for img in files:

        # open file
        hf = h5py.File(os.path.join(file_route + '\\', img), 'r')
        print('>>>> Opening file: {}'.format(os.path.join(out_path + img)))

        # control
        if cnt == 1 and plot_stats is True:
            for el in hf['Soil_Moisture_Retrieval_Data_3km']:
                print(el)
                cnt += 1

        # convert to table
        ppp = {'lat': np.array(hf['Soil_Moisture_Retrieval_Data_3km']['latitude_3km']),
               'lon': np.array(hf['Soil_Moisture_Retrieval_Data_3km']['longitude_3km']),
               'sm': np.array(hf['Soil_Moisture_Retrieval_Data_3km']['soil_moisture_3km'])}  # cm3/cm3
               # 'sm': np.array(hf['Soil_Moisture_Retrieval_Data_3km']['vegetation_water_content_3km'])}  # kg/m3

        # same data but with other structure
        # ppp = {'lat': np.array(hf['Soil_Moisture_Retrieval_Data']['latitude']),
        #       'lon': np.array(hf['Soil_Moisture_Retrieval_Data']['longitude']),
        #       'sm': np.array(hf['Soil_Moisture_Retrieval_Data']['soil_moisture'])}

        data = pd.DataFrame(data=ppp)
        del hf, ppp

        # filter empty/nan values ......
        filtered = data[(data['lat'] > -20.653346148076054) & (data['lat'] < -19.78738018198621)]
        filtered1 = filtered[(filtered['lon'] > -68.06579589843749) & (filtered['lon'] < -67.0330810546875)]
        filtered2 = filtered1[(filtered['sm'] > 0.0)]

        # print data
        if plot_stats is True:
            print('Stats image: {}'.format(img))
            print('Data set = \n', data, '\nshape = ', data.shape)
            print('Filtered Data set = \n', filtered2, '\nshape = ', filtered2.shape)

        # save memory
        del filtered, filtered1, data

        # plot
        fig, ax = plt.subplots()
        ax.set_xlabel('longitude')
        ax.set_ylabel('latitude')
        ax.plot(poly_lon, poly_lat, linewidth=1.0, color='r', label='salar de Uyuni')
        ax.plot(in_poly_lon, in_poly_lat, linewidth=1.0, color='orange', label='area of interested')
        ax.scatter(station_coordinates[0], station_coordinates[1], s=[50], label='gnss station', color='g', marker='^')
        ax.scatter(filtered2['lon'], filtered2['lat'], s=filtered2['sm']*10, color='b', label='soil moisture measurement')
        ax.set_title('Location Salar de Uyuni')
        ax.legend(fontsize=6, frameon=False, loc=9, ncol=4)  # loc
        plt.grid()

        # set axis for date
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
        plt.gcf().autofmt_xdate()

        plt.savefig(os.path.join(out_path, str(img)[0:len(img)-2]+'pdf'))
        plt.savefig(os.path.join(out_path, str(img)[0:len(img)-2]+'png'))
        if plot_data is True:
            plt.show()

        if len(filtered2['sm']) > 0:
            img_date = img[len('SMAP_L2_SM_AP_01065_D_'):len('SMAP_L2_SM_AP_01065_D_')+8]
            dic_data[img_date] = np.asarray(filtered2['sm']).tolist()

    with open(os.path.join(out_path, 'SMAP_L2_SM_AP_01065_D_.json'), 'w') as outfile:
        json.dump(dic_data, outfile)
    print('----> Dictionary was exported as json file')

    return dic_data


def read_h5_smap_l3(file_route: str,  # folder where the files are located
                    station_coordinates: np.ndarray,
                    outer_polygon: np.ndarray,
                    json_output_name: str,
                    output_folder: str = 'SMAP_L3_SM_P_E',
                    plot_data: bool = False,
                    print_stats: bool = False) -> Dict:
    """
    Read *.h5 data from the satellite mission:
      - SMAP L2 Radar/Radiometer Half-Orbit 9 km EASE-Grid Soil Moisture, Version 3 (SMAP_L3_SM_P_E)
        see: https://nsidc.org/data/SPL3SMP_E/versions/5
    Read h5-image and export the points inside a polygon
      - Create a polygon in: http://geojson.io/
    :param file_route: route to the folder where the sat data is stored
    :param station_coordinates: coordinates of the station
    :param outer_polygon: polygon (rectangle) coordinates of the area where search for data
    :param json_output_name: name of file to store the dic: 'SMAP_L3_SM_P_E_' + output_name + '.json'
    :param output_folder: name of the folder where the output files are stored
    :param plot_data: true if you want to show plots on screen
    :param print_stats: true if you want to print some comtrol statements (for developing)
    :return: dictionary with am and pm data and daily average
    """
    # ignore warnings in this function
    warnings.filterwarnings("ignore")

    # get files inside the input folder
    files = os.listdir(file_route + '\\')
    print('----> Reading satellite data from folder {}'.format(file_route))

    # create output folder
    out_path = os.path.join(output_folder + '_outputs\\')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if not os.path.exists(out_path + '\\json'):
        os.makedirs(out_path + '\\json')
    if not os.path.exists(out_path + '\\plots'):
        os.makedirs(out_path + '\\plots')

    if print_stats is True:
        print('Output structure was set -> main folder {}, plots folder {}, '
              'json folder {}'.format(out_path, out_path + '\\plots', out_path + '\\json'))

    # define polygons
    poly_lon, poly_lat = [], [],
    for i in range(0, len(outer_polygon)):
        poly_lon.append(outer_polygon[i][0])
        poly_lat.append(outer_polygon[i][1])

    date_am = []  # to store dates of measurements
    sm_values_am = []  # to store soil moisture values
    date_pm = []  # to store dates of measurements
    sm_values_pm = []  # to store soil moisture values
    date_day = []  # to store dates of measurements
    sm_values_day = []  # to store soil moisture values

    # salar shape for the plots
    salar_shape = get_salar_shape()

    # 2 open files
    for img in files:

        # open file
        hf = h5py.File(os.path.join(file_route + '\\', img), 'r')
        print('>>>> Opening file: {}'.format(os.path.join(out_path + img)))

        # convert to table AM data
        lat = np.array(hf['Soil_Moisture_Retrieval_Data_AM']['latitude'])
        lon = np.array(hf['Soil_Moisture_Retrieval_Data_AM']['longitude'])
        sm = np.array(hf['Soil_Moisture_Retrieval_Data_AM']['soil_moisture'])
        size = len(lat) * len(lat[0])

        data_am = {'lat': np.reshape(lat, (size,)),
                   'lon': np.reshape(lon, (size,)),
                   'sm': np.reshape(sm, (size,))}
        del lat, lon, sm, size

        # convert to table PM data
        lat = np.array(hf['Soil_Moisture_Retrieval_Data_PM']['latitude_pm'])
        lon = np.array(hf['Soil_Moisture_Retrieval_Data_PM']['longitude_pm'])
        sm = np.array(hf['Soil_Moisture_Retrieval_Data_PM']['soil_moisture_pm'])
        size = len(lat) * len(lat[0])

        data_pm = {'lat': np.reshape(lat, (size,)),
                   'lon': np.reshape(lon, (size,)),
                   'sm': np.reshape(sm, (size,))}
        del lat, lon, sm, size

        # convert dic to pandas
        data_sat_am = pd.DataFrame(data=data_am)
        data_sat_pm = pd.DataFrame(data=data_pm)
        del hf, data_pm, data_am

        # filter empty/nan values ......
        # TODO: UPDATE THIS VALUES FROM POLYGONS
        filtered = data_sat_am[(data_sat_am['lat'] > -20.653346148076054) & (data_sat_am['lat'] < -19.78738018198621)]
        filtered1 = filtered[(filtered['lon'] > -68.06579589843749) & (filtered['lon'] < -67.0330810546875)]
        filt_data_am = filtered1[(filtered['sm'] > 0.0)]
        del filtered, filtered1

        filtered = data_sat_pm[(data_sat_pm['lat'] > -20.653346148076054) & (data_sat_pm['lat'] < -19.78738018198621)]
        filtered1 = filtered[(filtered['lon'] > -68.06579589843749) & (filtered['lon'] < -67.0330810546875)]
        filt_data_pm = filtered1[(filtered['sm'] > 0.0)]
        del filtered, filtered1

        # plot
        fig, ax = plt.subplots()
        ax.set_xlabel('longitude')
        ax.set_ylabel('latitude')

        # station coordinates
        ax.scatter(station_coordinates[0], station_coordinates[1], s=[50], label='gnss station', color='g', marker='^')

        # area of study
        radio = np.rad2deg(5/4600)
        circle = Circle((station_coordinates[0], station_coordinates[1]), radio, facecolor='none',
                        edgecolor=(0, 1, 0), linewidth=3, alpha=0.5, label='circle R=5 Km')
        ax.add_patch(circle)

        # polygon where to search for data
        ax.plot(poly_lon, poly_lat, linewidth=1.0, color='r', label='salar de Uyuni')

        # salar shape
        ax.plot(salar_shape[0], salar_shape[1], label='Salar shape', alpha=0.5, color='orange')

        # soil moisture values
        ax.scatter(filt_data_am['lon'], filt_data_am['lat'], s=filt_data_am['sm']*10, color='b',
                   label='soil moisture measurement AM')
        ax.scatter(filt_data_pm['lon'], filt_data_pm['lat'], s=filt_data_pm['sm']*10, color='r',
                   label='soil moisture measurement PM')

        ax.set_title('Location Salar de Uyuni')
        ax.legend(fontsize=6, frameon=False, loc=9, ncol=4)  # loc
        plt.grid()

        op = out_path + '\\plots'
        plt.savefig(os.path.join(op, str(img)[0:len(img)-2]+'pdf'))
        plt.savefig(os.path.join(op, str(img)[0:len(img)-2]+'png'))

        if plot_data is True:
            plt.show()

        # get date
        img_date = img[len('SMAP_L3_SM_P_E_'):len('SMAP_L3_SM_P_E_') + 8]

        if len(filt_data_am['sm']) > 0:
            # get values am
            sm_data_am = np.concatenate([np.asarray(filt_data_am['sm'])]).tolist()
            date_am.append(img_date)
            sm_values_am.append(average_of_array(sm_data_am))

        if len(filt_data_pm['sm']) > 0:
            # get values am
            sm_data_pm = np.concatenate([np.asarray(filt_data_pm['sm'])]).tolist()
            date_pm.append(img_date)
            sm_values_pm.append(average_of_array(sm_data_pm))

        if len(filt_data_pm['sm']) > 0 or len(filt_data_am['sm']) > 0:
            # get values am
            sm_data_day = np.concatenate([np.asarray(filt_data_am['sm']), np.asarray(filt_data_pm['sm'])]).tolist()
            date_day.append(img_date)
            sm_values_day.append(average_of_array(sm_data_day))

    dic2 = {'am': {'dates_am': date_am, 'sm_mean_value_am': sm_values_am},
            'pm': {'dates_pm': date_pm, 'sm_mean_value_pm': sm_values_pm},
            'daily': {'dates_day': date_day, 'sm_mean_value_day': sm_values_day}}

    # route to export
    fn = os.path.join(out_path + '\\json', 'SMAP_L3_SM_P_E1_' + json_output_name + '.json')

    # to json
    with open(fn, 'w') as outfile:
        json.dump(dic2, outfile, indent=4)  # , separators=('\n', '='))
    print('----> Dictionary was exported as json file with mean soil moisture values: {}'.format(fn))
    # print(json.dumps(dic2, indent=4))

    return dic2


def read_json_files(route_to_json: str,
                    separator: str = '\\') -> Dict:
    """
    Join all json files inside a folder
    Export a json file with all data together
    :param route_to_json: folder name in which the json folder is located
    :param separator: define folder separator to the links -> win='\\', linux='/'
    :return: dictionary with all the satellite data from json file
    """

    # folder name
    folname = route_to_json + separator + 'json'
    print('----> Reading satellite data from json files in folder {}'.format(folname))

    # get file names inside a folder
    json_files = os.listdir(folname)

    data = {}  # dic to store sat data

    cnt = 1  # just a counter

    for js in json_files:

        # route to json file
        jfile = os.path.join(folname + separator, js)

        # open file
        with open(jfile) as f:
            dic = json.load(f)

            if cnt == 1:
                data = dic.copy()
                cnt += 1
            else:
                data['am']['dates_am'].extend(dic['am']['dates_am'])
                data['am']['sm_mean_value_am'].extend(dic['am']['sm_mean_value_am'])
                data['pm']['dates_pm'].extend(dic['pm']['dates_pm'])
                data['pm']['sm_mean_value_pm'].extend(dic['pm']['sm_mean_value_pm'])
                data['daily']['dates_day'].extend(dic['daily']['dates_day'])
                data['daily']['sm_mean_value_day'].extend(dic['daily']['sm_mean_value_day'])

    fn = os.path.join(route_to_json, 'SMAP_L3_SM_P_E1_resume.json')
    with open(fn, 'w') as outfile:
        json.dump(data, outfile, indent=4)
    print('----> join json file was exported to {}'.format(fn))
    # print(json.dumps(data, indent=4))

    return data


def plot_json_data(sat_data: Dict,
                   output_path: str,
                   plot_rain: bool = True,
                   plot_am: bool = True,
                   plot_pm: bool = True,
                   plot_mean: bool = True,
                   separator: str = '\\') -> bool:
    """
    Plot soil moisture data from satellite missions and climate data

    :param sat_data: satellite data as dictionary
    :param output_path: solder where to export results
    :param plot_rain: true if you want to add rain values (mm) to the plot
    :param plot_am: true if you want to plot soil moisture data am
    :param plot_pm: true if you want to plot soil moisture data pm
    :param plot_mean: true if you want to plot mean soil moisture data (am+pm / 2)
    :param separator: define folder separator to the links -> win='\\', linux='/'
    :return true if correct
    """

    data_am = sat_data['am']
    data_am = pd.DataFrame.from_dict(data_am)
    data_pm = sat_data['pm']
    data_pm = pd.DataFrame.from_dict(data_pm)
    data_d = sat_data['daily']
    data_d = pd.DataFrame.from_dict(data_d)

    # transform dates
    data_am['dates_am'] = pd.to_datetime(data_am['dates_am'], format='%Y%m%d')
    data_pm['dates_pm'] = pd.to_datetime(data_pm['dates_pm'], format='%Y%m%d')
    data_d['dates_d'] = pd.to_datetime(data_d['dates_day'], format='%Y%m%d')

    # plot
    fig, ax = plt.subplots()
    ax.set_title('Soil Moisture Salar de Uyuni')

    # soil moisture data
    if plot_am is True:
        ax.scatter(data_am['dates_am'], data_am['sm_mean_value_am'], label='Average AM', color='red')
    if plot_pm is True:
        ax.scatter(data_pm['dates_pm'], data_pm['sm_mean_value_pm'], label='Average PM', color='blue')
    if plot_mean is True:
        ax.scatter(data_d['dates_d'], data_d['sm_mean_value_day'], label='Average daily',
                   facecolors='none', edgecolors='green')

    ax.set_xlabel('Date')
    ax.set_ylabel('Satellite Soil Moisture cm3/cm3')

    if plot_rain is True:
        # read clima and filter
        clima = read_clima_data(plot_stats=False, export_plot=False)
        begin, end = min(data_d['dates_d']), max(data_d['dates_d'])
        clima1 = clima[(clima['date'] >= begin) & (clima['date'] <= end)]

        ax2 = ax.twinx()
        ax2.set_ylabel('Rain (mm)')
        ax2.bar(clima1['date'], clima1['Precipitación'], color='purple', label='rain', linewidth=2.0)
        # ax2.plot(clima1['date'], clima1['Precipitación'], color='g', label='rain', linewidth=1.0)

    fig.legend(loc='upper left')
    plt.grid()

    # set axis for date
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.gcf().autofmt_xdate()

    file_ = os.path.join(output_path + separator, 'SoilMoisturPlot.pdf')
    plt.savefig(file_)
    file_ = os.path.join(output_path + separator, 'SoilMoisturPlot.png')
    plt.savefig(file_)

    plt.show()

    return True


if __name__ == "__main__":

    # 1. read meteo data
    read_clima_data()

    # 2. get soil moisture from satellite information

    # gnss station coordinates (lon, lat)
    station_coord = np.array([-67.62740452, -20.24161967])

    # salar location (lon, lat)
    polygon = [[-68.06579589843749, -20.653346148076054],
               [-67.0330810546875, -20.653346148076054],
               [-67.0330810546875, -19.78738018198621],
               [-68.06579589843749, -19.78738018198621],
               [-68.06579589843749, -20.653346148076054]]

    # sat Band L3
    read_h5_smap_l3('SMAP_L3_SM_P_E', station_coord, np.asarray(inner_poly), np.asarray(polygon), 'dec_2016')
    json_data = read_json_files('SMAP_L3_SM_P_E_outputs')
    plot_json_data(json_data, 'SMAP_L3_SM_P_E_outputs', plot_rain=False)

"""
write something here!
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import os
from typing import List

# month = ['Jan', 'Feb', 'Mar', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan']
# date = ['20150101', '20150201', '20150301', '20150401', '20150501', '20150601',
#        '20150701', '20150801', '20150901', '20151001', '20151101', '20151201', '20160101']


def set_snippets():
    if not os.path.exists('../plots'):
        os.mkdir('../plots')


def plot_quicklook(metrics_success,
                   metrics_fail,
                   station_name: str,
                   freq: int,
                   ref_az: List[int] = [0, 360],
                   plot_az: List[int] = [0, 360],
                   peak2noise: float = 2.0,
                   ampl: float = 2.0
                   ):
    """
    PLot output of quicklook function
    * This function was taken and edited from homework02
    :param metrics_success: accepted values from the periodogram
    :param metrics_fail: rejected values from the periodogram
    :param station_name: station name (lower case)
    :param freq: frequency used for the plot, 0: All GPS freq, 1: GPS L1, 2: GPS L2
    :param ref_az: min/max azimuths you want to plot
    :param plot_az: plot these two azimuths values as vertical lines
    :param peak2noise: plot minimum peak2noise value to reject the observations
    :param ampl: plot minimum amplitude value to reject the observations
    """

    if type(station_name) != str:
        raise ValueError('Error station name: expected entry: string')
    else:
        station_name = station_name.lower()

    if ref_az[0] < 0 or ref_az[1] < 0 or plot_az[0] < 0 or plot_az[0] < 0:
        raise ValueError('Expected azimuth values >0')
    elif ref_az[0] > 360 or ref_az[1] > 360 or plot_az[0] > 360 or plot_az[0] > 360:
        raise ValueError('Expected azimuth values <360')

    # metrics
    metrics_fail = metrics_fail[metrics_fail['Azimuth'].between(ref_az[0], ref_az[1])]
    metrics_success = metrics_success[metrics_success['Azimuth'].between(40, 320)]

    if freq == 0:
        fr = 'All GPS Frequencies'
    elif freq == 1:
        fr = 'GPS L1'
    elif freq == 2:
        fr = 'GPS L2'
    else:
        raise ValueError('Unknown Frequency')

    avg_rh = np.mean(metrics_success['Reflector Height'])
    print(f'Average reflector height value: {avg_rh:.1f}')

    # plotting the qc metrics graphs
    fig, axes = plt.subplots(ncols=1, nrows=3, figsize=(10, 10), sharex=True)
    fig.suptitle(f'QuickLook Retrieval Metrics: {station_name} {fr}', size=16)

    cnt = 1
    for i, ax in enumerate(axes):
        g = sns.scatterplot(x='Azimuth', y=metrics_success.columns[i + 1], data=metrics_success, ax=ax, label='good')
        g = sns.scatterplot(x='Azimuth', y=metrics_fail.columns[i + 1], data=metrics_fail,
                            ax=ax, color='lightgrey', label='bad')
        ax.axvline(plot_az[0], label='min azimuth', linewidth=0.7, color='green')
        ax.axvline(plot_az[1], label='max azimuth', linewidth=0.7, color='orange')
        if cnt == 1:
            avg_rh
            ax.axhline(avg_rh, color='red', linewidth=0.7, label='mean h')
            ax.legend(loc=8, ncol=5)
        elif cnt == 2:
            ax.axhline(peak2noise, color='red', linewidth=0.7)
            ax.legend(loc=8, ncol=4)
        elif cnt == 3:
            ax.axhline(ampl, color='red', linewidth=0.7)
            ax.legend(loc=8, ncol=4)
        cnt += 1
    plt.tight_layout()
    plt.show()
    

def conv_json_to_pd(json_data: dict):
    """
    Convert read date from json files into a pandas data structure
    :param json_data: content of a json file
    :return
    """
    data_am = json_data['am']
    data_am = pd.DataFrame.from_dict(data_am)
    data_pm = json_data['pm']
    data_pm = pd.DataFrame.from_dict(data_pm)
    data_d = json_data['daily']
    data_d = pd.DataFrame.from_dict(data_d)

    # transform dates
    data_am['dates_am'] = pd.to_datetime(data_am['dates_am'], format='%Y%m%d')
    data_pm['dates_pm'] = pd.to_datetime(data_pm['dates_pm'], format='%Y%m%d')
    data_d['dates_d'] = pd.to_datetime(data_d['dates_day'], format='%Y%m%d')

    return data_am, data_pm, data_d


def soil_mois_analysis(daily_average: pd,
                       clima_data: pd,
                       soil_moisture_info: dict):
    """
    Plot the final results in different plots
    :param daily_average: daily average of amplitudes and heights as pandas
    :param clima_data: clima information as panda
    :param soil_moisture_info: validation data from smap l3 as a dictionary (output satetllite_data-lib/read_json_files)
    """

    # get soil moisture data
    data_am, data_pm, data_mean = conv_json_to_pd(soil_moisture_info)

    # transform dates
    daily_average['dates'] = pd.to_datetime(daily_average['dates'], format='%Y-%m-%d')

    # filter data sets
    begin, end = min(data_mean['dates_d']), max(data_mean['dates_d'])
    clima1 = clima_data[(clima_data['date'] >= begin) & (clima_data['date'] <= end)]
    daily_average1 = daily_average[(daily_average['dates'] >= begin) & (daily_average['dates'] <= end)]

    # get phases
    ph = get_phase1(daily_average1['rh'], 'L1')

    def plot_clima():
        begin1 = pd.to_datetime('20160201', format='%Y-%m-%d')
        end1 = pd.to_datetime('20160301', format='%Y-%m-%d')
        clima2 = clima_data[(clima_data['date'] >= begin1) & (clima_data['date'] <= end1)]
        fig, ax = plt.subplots()
        ax.set_title('Salar de Uyuni: März Klima')
        ax.set_xlabel('Date')
        ax.set_ylabel('Regen (mm)')
        ax.bar(clima2['date'], clima2['Precipitación'], color='purple', label='rain', linewidth=2.0)

        ax1 = ax.twinx()
        data_am1 = data_am[(data_am['dates_am'] >= begin1) & (data_am['dates_am'] <= end1)]
        ax1.scatter(data_am1['dates_am'], data_am1['sm_mean_value_am'], label='SMAP-Average AM', color='red',
                   alpha=0.5)

        ax3 = ax.twinx()
        daily_average2 = daily_average[(daily_average['dates'] >= begin1) & (daily_average['dates'] <= end1)]
        ap3, = ax3.plot(daily_average2['dates'], daily_average2['amp'], color='orange', label='Amplitude')

        # make axis beauti
        ax3.spines['right'].set_position(('axes', 1.089))
        ax3.yaxis.label.set_color(ap3.get_color())
        ax3.tick_params(axis='y', colors=ap3.get_color())        
        ax1.yaxis.label.set_color('red')
        ax.tick_params(axis='y', colors='purple')
        
        # set axis for date
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        # plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
        plt.gcf().autofmt_xdate()

        fig.legend(loc='upper left')
        plt.grid()
        
    def plot_sm():
        """
        from paper -> [5]
        For uniform soil moisture profiles, soil moisture and A_mpi have a linear inverse relationship

        from paper -> [1]
        - whereas the freq of SNR oscillations depends on surface-antenna geometry, teh amplitude of SNR oscillations
          depends largely on surface reflectivity.
        - when a fixed area of the ground with unchanging soil composition, variation in the amplitude of snr oscillations
          can serve a proxy for changes in near surface soil moisture
        - ... removing the Ad reveals multipath oscillations whose amplitude is proportional to the surface reflectivity
        """

        fig, ax = plt.subplots()
        ax.set_title('GNSS-IR L1: Amplitude')

        ax.set_xlabel('Date')
        ax.set_ylabel('Satellite Soil Moisture cm3/cm3')

        # soil moisture data
        plot_am, plot_pm, plot_mean = False, False, True

        if plot_am is True:
            ax.scatter(data_am['dates_am'], data_am['sm_mean_value_am'], label='SMAP-Average AM', color='red',
                       alpha=0.5)
        if plot_pm is True:
            ax.scatter(data_pm['dates_pm'], data_pm['sm_mean_value_pm'], label='SMAP-Average PM', color='blue')
        if plot_mean is True:
            ax.scatter(data_mean['dates_d'], data_mean['sm_mean_value_day'], label='SMAP-Average daily',
                       facecolors='none', edgecolors='green')

        # plot rain
        ax2 = ax.twinx()
        ax2.set_ylabel('Rain (mm)')
        ax2.bar(clima1['date'], clima1['Precipitación'], color='purple', label='rain', linewidth=2.0)

        # plot gnss-ir amplitudes
        ax3 = ax.twinx()
        ax3.set_ylabel('Am (v/v)')
        ap3, = ax3.plot(daily_average1['dates'], daily_average1['amp'], color='orange', label='Amplitude')
        ax3.invert_yaxis()
        ax.set_xlim([begin, end])
        
        # make axis beauti
        ax3.spines['right'].set_position(('axes', 1.1))        
        ax3.yaxis.label.set_color(ap3.get_color())
        ax3.tick_params(axis='y', colors=ap3.get_color())        
        ax2.yaxis.label.set_color('purple')
        ax2.tick_params(axis='y', colors='purple')        
        ax.yaxis.label.set_color('green')
        ax.tick_params(axis='y', colors='green')
        
        # set axis for date
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        # plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
        plt.gcf().autofmt_xdate()

        fig.legend(loc='lower right')
        plt.grid()
        plt.tight_layout()
        plt.savefig('sm_amp.png', dpi=250, transparent=False)
        plt.savefig('sm_amp.pdf', dpi=250, bbox_inches = 'tight')


    def plot_reflector_depth():
        """
        from paper -> [5]
        For uniform moisture profiles, as the soil becomes wetter, the height estimated from
        the Lomb–Scargle periodogram (Heff ) decreases
        """

        fig, ax = plt.subplots()
        ax.set_title('GNSS-IR L1: Reflected Height')

        ax.set_xlabel('Date')
        ax.set_ylabel('Satellite Soil Moisture cm3/cm3')

        # soil moisture data
        plot_am, plot_pm, plot_mean = False, False, True

        if plot_am is True:
            ax.scatter(data_am['dates_am'], data_am['sm_mean_value_am'], label='SMAP-Average AM', color='red',
                       alpha=0.5)
        if plot_pm is True:
            ax.scatter(data_pm['dates_pm'], data_pm['sm_mean_value_pm'], label='SMAP-Average PM', color='blue')
        if plot_mean is True:
            ax.scatter(data_mean['dates_d'], data_mean['sm_mean_value_day'], label='SMAP-Average daily',
                       facecolors='none', edgecolors='green')

        # plot rain
        ax2 = ax.twinx()
        ax2.set_ylabel('Rain (mm)')
        ax2.bar(clima1['date'], clima1['Precipitación'], color='purple', label='rain', linewidth=2.0)

        # plot depth
        mean_depth = sum(daily_average['rh']) / len(daily_average['rh'])
        print('mean_depth value = {}'.format(mean_depth))
        diff_h = (np.asarray(daily_average['rh']) - mean_depth) * 100  # in cm

        # height from periodrogram
        ax3 = ax.twinx()
        ax3.set_ylabel('Effective Height (m)')
        ax3.plot(daily_average1['dates'], daily_average1['rh']-18.0, color='orange', label='Effective Height')
        # ax3.invert_yaxis()
        
        # make axis beauti
        ax3.spines['right'].set_position(('axes', 1.089))
        ax3.yaxis.label.set_color('orange')
        ax3.tick_params(axis='y', colors='orange')        
        ax2.yaxis.label.set_color('purple')
        ax.tick_params(axis='y', colors='green')
        ax.yaxis.label.set_color('green')

        # set axis for date
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        # plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=15))
        plt.gcf().autofmt_xdate()
        ax.set_xlim([begin, end])

        # use of a float for the position:
        fig.legend(loc='lower right')
        plt.grid()
        plt.tight_layout()
        plt.savefig('RH.png')
        plt.savefig('RH.pdf', dpi=250, bbox_inches='tight')

    def plot_clima1():
        begin1 = pd.to_datetime('20160201', format='%Y-%m-%d')
        end1 = pd.to_datetime('20160301', format='%Y-%m-%d')
        clima2 = clima_data[(clima_data['date'] >= begin1) & (clima_data['date'] <= end1)]
        fig, ax = plt.subplots()
        ax.set_title('Salar de Uyuni: März Klima')
        ax.set_xlabel('Date')
        ax.set_ylabel('Regen (mm)')
        ax.bar(clima2['date'], clima2['Precipitación'], color='purple', label='rain', linewidth=2.0)

        ax1 = ax.twinx()
        data_am1 = data_am[(data_am['dates_am'] >= begin1) & (data_am['dates_am'] <= end1)]
        ax1.scatter(data_am1['dates_am'], data_am1['sm_mean_value_am'], label='SMAP-Average AM', color='red',
                    alpha=0.5)

        ax3 = ax.twinx()
        daily_average2 = daily_average[(daily_average['dates'] >= begin1) & (daily_average['dates'] <= end1)]
        ap3, = ax3.plot(daily_average2['dates'], daily_average2['rh'], color='orange', label='Effective Height')

        # make axis beauti
        ax3.spines['right'].set_position(('axes', 1.1))
        ax3.yaxis.label.set_color(ap3.get_color())
        ax3.tick_params(axis='y', colors=ap3.get_color())        
        ax1.yaxis.label.set_color('purple')
        ax.tick_params(axis='y', colors='blue')

        
        # set axis for date
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()
        ax.set_xlim([begin1, end1])

        fig.legend(loc='upper left')
        plt.grid()

    def plot_phase_smm():

        # filter data -> no efficient but works!
        ph1 = []
        sm1 = []
        cntr1 = 0
        for el in data_am['dates_am']:
            cntr = 0
            #if el not in daily_average1['dates']:
            #    print('WARNING: this date is not in the data set', el)
            #else:
            #    for el1 in daily_average1['dates']:
            #        if el == el1:
            #            ph1.append(ph[cntr])
            #            # print('CONTROL = ', len(ph1), el, el1, ph[cntr])
            #            sm1.append(data_am['sm_mean_value_am'][cntr1])
            #            break
            #        cntr += 1
            for el1 in daily_average1['dates']:
                if el == el1:
                    ph1.append(ph[cntr])
                    sm1.append(data_am['sm_mean_value_am'][cntr1])
                    break
                cntr += 1
            cntr1 += 1
        # print('LENGTHS CONTROL = ', len(ph1), len(sm1))

        fig, ax = plt.subplots()
        ax.set_title('Salar de Uyuni: Soil Moisture vs Phase')

        ax.set_xlabel('smc (cm3/cm3)')
        ax.set_ylabel('phase (rad)')

        ax.scatter(sm1, ph1)
        fig.legend(loc='upper left')
        plt.grid()
    
    def plot_phase_new():
        
        def todate(doy):
            """ valid only for 2016 xd """
            # dates        j   f   m   a   m  j    j   a   s   o   n   d 
            days_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            month = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

            cnt = 0
            control = False
            cumsum = days_month[0]
                        
            while control==False:
                if cumsum < doy:
                    cnt += 1
                    cumsum += days_month[cnt]
                else:
                    control = True
            
            if cnt > 0:
                d = doy - sum(days_month[0:cnt])
            else:
                d = doy
            
            if cnt <= 8:
                date = '20160' + str(cnt+1)
            else:
                date = '2016' + str(cnt+1)
            
            if d <= 9:
                date += '0' + str(d)
            else:
                date += str(d)
            
            return pd.to_datetime(date, format='%Y%m%d')
            
        
        data_am1 = data_am[(data_am['dates_am'] >= '01-01-2016')]
        dff = pd.read_csv("../Files/amde_average_phase.csv.txt", sep=",", header=None)        
        
        dd, phd = [], []
        for i in range(0, len(dff[2])):
            dd.append(todate(dff[2][i]))
            phd.append(dff[3][i])
        print(len(dff[2]), len(dff[3]), len(dd), len(phd))
        
        # ph1, sm1 = [], []
        # cntr1 = 0
        # for el in data_am['dates_am']:
            # cntr = 0
            # for el1 in dd:
                # if el == el1:
                    # ph1.append(phd[cntr])
                    # sm1.append(data_am['sm_mean_value_am'][cntr1])
                    # break
                # cntr += 1
            # cntr1 += 1
        
        #fig, ax = plt.subplots()
        #ax.scatter(sm1, ph1)
        #plt.show()
        
        fig, ax = plt.subplots()
        ax.set_title('Salar de Uyuni: Soil Moisture Analysis')

        ax.set_xlabel('Date')
        ax.set_ylabel('Satellite Soil Moisture cm3/cm3')

        # soil moisture data
        plot_am, plot_pm, plot_mean = False, False, True

        if plot_am is True:
            ax.scatter(data_am['dates_am'], data_am['sm_mean_value_am'], label='SMAP-Average AM', color='red',
                       alpha=0.5)
        if plot_pm is True:
            ax.scatter(data_pm['dates_pm'], data_pm['sm_mean_value_pm'], label='SMAP-Average PM', color='blue')
        if plot_mean is True:
            ax.scatter(data_mean['dates_d'], data_mean['sm_mean_value_day'], label='SMAP-Average daily',
                       facecolors='none', edgecolors='green')

        # plot rain
        ax2 = ax.twinx()
        ax2.set_ylabel('Rain (mm)')
        ax2.bar(clima1['date'], clima1['Precipitación'], color='purple', label='rain', linewidth=2.0)

        # plot depth
        mean_depth = sum(daily_average['rh']) / len(daily_average['rh'])
        print('mean_depth value = {}'.format(mean_depth))
        diff_h = (np.asarray(daily_average['rh']) - mean_depth) * 100  # in cm

        # height from periodrogram
        ax3 = ax.twinx()
        ax3.set_ylabel('phase (deg)')
        ap3, = ax3.plot(dd, phd, color='orange', label='Phase')
        ax3.invert_yaxis()
        
        # make axis beauti
        ax3.spines['right'].set_position(('axes', 1.089))
        ax3.yaxis.label.set_color(ap3.get_color())
        ax3.tick_params(axis='y', colors=ap3.get_color())        
        ax2.yaxis.label.set_color('purple')
        ax.tick_params(axis='y', colors='green')
        ax.yaxis.label.set_color('green')

        # set axis for date
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()
        ax.set_xlim([begin, end])

        # use of a float for the position:
        fig.legend(loc='lower right')
        plt.grid()

    plot_reflector_depth()
    # plot_clima1()
    plot_sm()
    # plot_clima()
    # plot_phase_smm()
    plot_phase_new()

    # TODO: https://matplotlib.org/3.4.3/gallery/ticks_and_spines/multiple_yaxis_with_spines.html


def get_phase(rh, gnss_freq, el_angle=5.0):

    if gnss_freq is 'L1':
        lw = 0.1930  # length wave l1 (m)
        #
    elif gnss_freq is 'L2':
        lw = 0.2442  # length wave l2 (m)
    else:
        raise ValueError('Unknown GNSS frequency!')

    sine = np.deg2rad(el_angle)
    # f = 4 * sine * np.pi / lw
    f = 2 * sine / lw

    return f * rh


def get_phase1(rh, gnss_freq, el_angle=5.0):
    """
    Test fucntion to get the amplitudes
    It is not working! 
    
    :param rh: reflected hights
    :param gnss_freq: gnss frequency to process (L1 or L2)
    :param el_angle: average elevation angle 
    """
    if gnss_freq is 'L1':
        lw = 0.1930  # length wave l1 (m)
        #
    elif gnss_freq is 'L2':
        lw = 0.2442  # length wave l2 (m)
    else:
        raise ValueError('Unknown GNSS frequency!')

    sine = np.deg2rad(el_angle)
    # f = 4 * sine * np.pi / lw
    f = 2 * sine / lw
    print(f)

    # return f * rh

    val = f * rh

    for el in val:
        while el > 2 * np.pi:
            el -= 2 * np.pi
        # print(el)

    r = []
    for i in range(0, len(val)):
        while val[i] > 2 * np.pi:
            val[i] -= 2 * np.pi
        r.append(val[i])
    
    # print('phases = \n', r)

    return np.rad2deg(r)

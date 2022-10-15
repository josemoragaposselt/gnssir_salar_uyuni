import os
import sys
import platform
import shutil
import urllib.request
import tarfile
import datetime
import pandas as pd
import re


def check_environment():
    try:
        os.environ['ORBITS']
        os.environ['REFL_CODE']
        os.environ['EXE']

        environment_set = True
    except KeyError:
        environment_set = False

    return environment_set


def set_environment(orbits='../orbits', refl_code='../', exe='../bin/exe'):
    os.environ['ORBITS'] = os.path.abspath(os.path.join(orbits))
    os.environ['REFL_CODE'] = os.path.abspath(os.path.join(refl_code))
    os.environ['EXE'] = os.path.abspath(os.path.join(exe))

    print('environment variable ORBITS set to path', os.environ['ORBITS'],
          '\nenvironment variable REFL_CODE set to path', os.environ['REFL_CODE'],
          '\nenvironment variable EXE set to path', os.environ['EXE'])


def download_crx2rnx(system=None, path_to_executables=None):
    if not system:
        system = platform.platform().lower()

    if not path_to_executables:
        # if no path is provided, then the path to executables will be in the repo from the notebooks
        try:
            path_to_executables = os.environ['EXE']
        except KeyError:
            print('you must have your environment variables set. Please use the set_environment function to set the your EXE environment variable path')
            sys.exit()
    else:
        os.environ['EXE'] = path_to_executables
        print(f'reset environment variable to the requested path_to_executables: {os.environ["EXE"]}')

    html_path = 'https://terras.gsi.go.jp/ja/crx2rnx.html'
    url = urllib.request.urlopen(html_path)
    html = str(url.read())

    regex = '(crx2rnx.)(RNXCMP_[\d|\w|_|.]+)'
    matches = re.finditer(regex, html)

    download_files = []
    for match in matches:
        file = match.group(2)
        download_files.append(file)
    os_types = {'macos': ['macos'], 'linux': ['64', '32'], 'sunos': ['sunos'], 'windows': ['64', '32']}
    matched = 0
    for os_type in os_types:
        if os_type in system:
            matched += 1
            if len(os_types[os_type]) > 1:
                for sub_type in os_types[os_type]:
                    str_match = [s for s in download_files if os_type in s.lower() and sub_type in s.lower()]
                    download_file = str_match[0]
                    downloaded_dir = download_file.replace(".tar.gz",'').replace(".zip",'')
            else:
                str_match = [s for s in download_files if os_type in s.lower()]
                download_file = str_match[0]
                downloaded_dir = download_file.replace(".tar.gz", '').replace(".zip", '')

    if matched == 0:
        print('We could not identify your operating system.'
              'Please add the parameter system= with either macos, linux32, linux64, sunos, windows32, or windows64')
        sys.exit()

    try:
        # download the correct file from the site
        download_path = 'https://terras.gsi.go.jp/ja/crx2rnx'
        print('downloading CRX2RNX file')
        urllib.request.urlretrieve(os.path.join(download_path, download_file),
                                   os.path.join(path_to_executables, download_file))

        # uncompress the file
        tar = tarfile.open(os.path.join(path_to_executables, download_file))
        tar.extractall(path_to_executables)
        tar.close()

        print('file placed in path to execubles set by EXE environment variable')
        current_destination = os.path.join(path_to_executables, f'{downloaded_dir}/bin/CRX2RNX')
        os.replace(current_destination, os.path.join(path_to_executables, 'CRX2RNX'))

        # cleanup and move the file where we need it
        os.remove(os.path.join(path_to_executables, download_file))
        shutil.rmtree(os.path.join(path_to_executables, downloaded_dir))

        print('finished')

    except FileNotFoundError:
        print('could not find the correct executable path. Please set the parameter path_to_excecutables= and set this to '
              'the absolute path to where you would like this file place. Please make sure this path is the same as your environment variable EXE')
        sys.exit()


def read_rh_files(filepath):

    # regex = '^ (?P<year>[ \d]+) +(?P<doy>[\d]+) +(?P<rh>[\d|-|.]+) +(?P<rh-amp>[\d|-|.]+)'
    regex = '^ (?P<year>[ \d]+) +(?P<doy>[\d]+) +(?P<rh>[\d|-|.]+) +(?P<nv>[\d]+) +(?P<mn>[\d]+) ' \
            '+(?P<d>[\d]+) +(?P<rhs>[\d|-|.]+) +(?P<rha>[\d|-|.]+)'

    data = {'dates': [], 'rh': [], 'amp': []}
    # read daily average reflector heights
    with open(filepath, 'r') as myfile:
        file = myfile.read()
        matches = re.finditer(regex, file, flags=re.MULTILINE)

        for match in matches:
            ydoy = f'{int(match.group("year"))}-{int(match.group("doy"))}'
            date = datetime.datetime.strptime(ydoy, '%Y-%j').date()
            data['dates'].append(date)
            data['rh'].append(float(match.group('rh')))
            data['amp'].append(float(match.group('rha')))

    return data


def read_subdaily(filepath):
    regex = '^ ?(?P<year>[ \d]+) +(?P<doy>[\d]+) +(?P<rh>[\d|-|.]+) +(?P<satellite>[\d]+) +(?P<UTCtime>[\d|.]+)'
    data = {'dates': [], 'rh': [], 'utctime': []}
     # read daily average reflector heights
    with open(filepath, 'r') as myfile:
        file = myfile.read()
        matches = re.finditer(regex, file, flags=re.MULTILINE)

        for match in matches:
            ydoy = f'{int(match.group("year"))}-{int(match.group("doy"))}'
            date = datetime.datetime.strptime(ydoy, '%Y-%j').date()
            data['dates'].append(date)
            data['rh'].append(float(match.group('rh')))
            data['datetime'].append(float(match.group('UTCtime')))
    return data


def quicklook_metrics(datakeys):
    quadrants = ['NW', 'NE', 'SW', 'SE']

    # re-organizing the data in a plotting friendly format
    success_data = {'Azimuth': [], 'Reflector Height': [], 'Peak to Noise': [], 'Amplitude': []}
    fail_data = {'Azimuth': [], 'Reflector Height': [], 'Peak to Noise': [], 'Amplitude': []}

    for i, quadrant in enumerate(quadrants):
        for j in datakeys[quadrant].keys():
            success_data['Azimuth'].append(datakeys[quadrant][j][0])
            success_data['Reflector Height'].append(datakeys[quadrant][j][1])
            success_data['Peak to Noise'].append(datakeys[quadrant][j][5])
            success_data['Amplitude'].append(datakeys[quadrant][j][4])
        for k in datakeys[f'f{quadrant}'].keys():
            fail_data['Azimuth'].append(datakeys[f'f{quadrant}'][k][0])
            fail_data['Reflector Height'].append(datakeys[f'f{quadrant}'][k][1])
            fail_data['Peak to Noise'].append(datakeys[f'f{quadrant}'][k][5])
            fail_data['Amplitude'].append(datakeys[f'f{quadrant}'][k][4])

    return pd.DataFrame(success_data), pd.DataFrame(fail_data)


if __name__ == '__main__':
    download_crx2rnx()
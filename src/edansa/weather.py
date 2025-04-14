''' This script is used to generate weather data for the NNA dataset.
'''
import glob

from pathlib import Path
import csv
import random
import json
import datetime

import pandas as pd
import pandas as pd
import numpy as np

TIMESTAMP_FORMAT = '%Y-%m-%d_%H:%M:%S'
ADELE_WEATER_DATA_FREQ = 3600 * 3  # 3 hours
MERRA_DATA_FREQ = 3600  # 1 hour
CORRECT_SHIFT_HOURS = -18

LOCAL = False
SCRATCH = 'scratch/enis/data/nna/'
LOCAL_PARENT = '/Users/berk/Documents/'
if LOCAL:
    WEATHER_DATA_FOLDER = f'{LOCAL_PARENT}{SCRATCH}weather_data/2017_2020'
    FILE_DATABASE = f'{LOCAL_PARENT}{SCRATCH}database/allFields_dataV10.pkl'
    NEON_WEATHER_DATA_FOLDER = (f'{LOCAL_PARENT}{SCRATCH}/' +
                                'weather_data/NEON_precipitation/')
    MERRA_WEATHER_DATA_FOLDER = f'{LOCAL_PARENT}{SCRATCH}weather_data/merra'
else:
    # SCRIPTS_DIR = '/home/enis/projects/nna/src/scripts/'
    WEATHER_DATA_FOLDER = f'/{SCRATCH}weather_data/2017_2020'
    FILE_DATABASE = f'/{SCRATCH}database/allFields_dataV10.pkl'
    NEON_WEATHER_DATA_FOLDER = f'/{SCRATCH}/weather_data/NEON_precipitation/'
    NEON_WIND_DATA_FOLDER = f'/{SCRATCH}weather_data/NEON_wind-2d/'
    MERRA_WEATHER_DATA_FOLDER = f'/{SCRATCH}weather_data/merra'
# os.chdir(SCRIPTS_DIR)

ROOT_PATH = f'{SCRATCH}labeling/samples'

# 40 Prudhoe or ANWR monitoring sites AND the Ivvavik sites
SHORT_INPUT_CSV_HEADERS = [
    'day_length', 'air_temp', 'snow_depth', 'cloud_fraction',
    'relative_humidity', 'runoff', 'rain_precip', 'snow_precip',
    'wind_direction', 'wind_speed'
]
# for Dalton and Dempster
LONG_INPUT_CSV_HEADERS = [
    'day_length', 'air_temp', 'snow_depth', 'cloud_fraction',
    'relative_humidity', 'runoff', 'rain_precip', 'snow_precip', 'total_precip',
    'wind_direction', 'wind_speed', 'snow_blowing_ground', 'snow_blowing_air'
]

SHORT_LOCATIONS = ('prudhoe', 'ivvavik', 'anwr')
LONG_LOCATIONS = ('dalton', 'dempster')

EXCELL_ALL_HEADERS = [
    'data_version', 'Annotator', 'File Name', 'Date', 'Start Time', 'End Time',
    'Length', 'Clip Path', 'Comments', 'weather_timestamp', 'region',
    'location', 'day_length', 'air_temp', 'snow_depth', 'cloud_fraction',
    'relative_humidity', 'runoff', 'rain_precip', 'rain_precip_mm',
    'snow_precip', 'wind_direction', 'wind_speed'
]


def load_all_adele_weather_data(  # pylint: disable=dangerous-default-value
        weather_data_folder=WEATHER_DATA_FOLDER,
        short_locations=SHORT_LOCATIONS,
        long_locations=LONG_LOCATIONS,
        short_input_csv_headers=SHORT_INPUT_CSV_HEADERS,
        long_input_csv_headers=LONG_INPUT_CSV_HEADERS,
        log=True,
        timezone='America/Anchorage',
        correct_timestamps=False):
    if timezone != 'America/Anchorage':
        raise NotImplementedError('Only Alaska time zone is supported')
    # print('weather_data_folder:', weather_data_folder)
    station_csv = csv_path_per_regloc(weather_data_folder)
    weather_df = []
    # print(station_csv)
    for (region, location), fname in station_csv.items():
        if log:
            print('region:', region, 'location:', location, 'fname:', fname)
        weather_rows = load_adele_weather_data(
            region,
            location,
            fname,
            short_locations,
            long_locations,
            short_input_csv_headers,
            long_input_csv_headers,
            correct_timestamps=correct_timestamps)
        weather_df.append(weather_rows)

    # comboine weather dataframes into one
    weather_rows = pd.concat(weather_df)
    return weather_rows


def fix_format_met_data(df):
    # Create a timestamp column
    df['timestamp'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    # move the timestamp one hour back to match the audio files
    # because the weather data is recorded at the end of the hour
    df['timestamp'] = df['timestamp'] - pd.Timedelta(hours=1)
    df['start_date_time'] = df['timestamp']
    df['end_date_time'] = df['timestamp'] + pd.Timedelta('1h')
    # Split the 'id' column into two new columns 'region' and 'location'
    # based on alphabetical and numeric part
    df[['region', 'location']] = df['id'].str.extract('([A-Za-z]+)([0-9]+)')
    delete = [
        'id',
        'year',
        'month',
        'day',
        'hour',
    ]
    df.drop(delete, axis=1, inplace=True)
    df = df[[
        'region',
        'location',
        'timestamp',
        'start_date_time',
        'end_date_time',
        'air_temp(C)',
        'rel_humid(%)',
        'rain_precip(mm/hr)',
        'snow_precip(mm/hr)',
        'total_precip(mm/hr)',
        'wind_spd(m/s)',
        'wind_dir(degTrue)',
        'incoming_solar(W/m^2)',
        'day_length(days)',
    ]].copy()
    df['region'] = df['region'].str.lower()
    df['location'] = df['location'].str.lower()
    df = df.rename(
        columns={
            'air_temp(C)': 'air_temp_C',
            'rel_humid(%)': 'relative_humidity_percent',
            'rain_precip(mm/hr)': 'rain_precip_mm_1hr',
            'snow_precip(mm/hr)': 'snow_precip_mm_1hr',
            'total_precip(mm/hr)': 'total_precip_mm_1hr',
            'wind_spd(m/s)': 'wind_speed_m_s',
            'wind_dir(degTrue)': 'wind_direction_deg',
            'incoming_solar(W/m^2)': 'incoming_solar_W_m2',
            'day_length(days)': 'day_length_days'
        })

    return df


def load_all_era5_data(csv_file_path, local_timezone, utc_timezone='UTC'):
    era5_data_df = pd.read_csv(csv_file_path)

    def add_sinp(x):
        if x < 10:
            return f'sinp0{x}'
        if x == 10:
            return f'sinp{x}'
        else:
            return f'{x}'

    def pick_region(x):
        if x < 11:
            return 'ivvavik'
        elif x < 31:
            return 'prudhoe'
        else:
            return 'anwr'

    df = era5_data_df.copy()
    df['timestamp'] = pd.to_datetime(df['Time'])
    df['timestamp'] = df['timestamp'].dt.tz_localize(
        utc_timezone).dt.tz_convert(local_timezone).dt.tz_localize(None)
    df['start_date_time'] = df['timestamp']
    df['end_date_time'] = df['timestamp'] + pd.Timedelta('1h')

    df['location'] = df['site'].apply(lambda x: add_sinp(x))
    df['region'] = df['site'].apply(lambda x: pick_region(x))

    df['longitude'] = df['.geo'].apply(
        lambda x: json.loads(x)['coordinates'][0])
    df['latitude'] = df['.geo'].apply(lambda x: json.loads(x)['coordinates'][1])

    df.rename(columns={'Windspeed': 'wind_speed_m_s'}, inplace=True)
    df.drop(columns=[
        'Time', 'site', 'system:index', 'air_temperature', 'skin_temperature',
        'system:time_start', '.geo'
    ],
            inplace=True)

    df = df[[
        'region', 'location', 'longitude', 'latitude', 'start_date_time',
        'end_date_time', 'timestamp', 'wind_speed_m_s'
    ]].copy()
    df.sort_values(by=['region', 'location', 'start_date_time'], inplace=True)
    return df


def load_all_merra_data(
    data_folder=MERRA_WEATHER_DATA_FOLDER,
    log=False,
    timezone='America/Anchorage',
):
    if timezone != 'America/Anchorage':
        raise NotImplementedError('Only Alaska timezone is supported')

    headers = [
        'id', 'year', 'month', 'day', 'hour', 'day_length(days)', 'air_temp(C)',
        'rel_humid(%)', 'rain_precip(mm/hr)', 'snow_precip(mm/hr)',
        'total_precip(mm/hr)', 'wind_spd(m/s)', 'wind_dir(degTrue)',
        'incoming_solar(W/m^2)'
    ]
    all_df = []
    _ = print(data_folder) if log else None
    for region_path in glob.glob(f'{data_folder}/*'):
        _ = print(region_path) if log else None
        region_csv = pd.read_csv(region_path,
                                 header=None,
                                 names=headers,
                                 skiprows=1)
        region_csv = fix_format_met_data(region_csv,)
        regions = (region_csv['region'].unique().tolist())  # type: ignore
        if len(regions) == 1 and regions[0] == 'ivvavik':
            region_csv['location'] = 'sinp' + region_csv['location']
        if log:
            print(regions)
            print(region_csv['location'].unique())  # type: ignore

        all_df.append(region_csv)
    merra_data = pd.concat(all_df)
    merra_data.sort_values(by=['region', 'location', 'timestamp'], inplace=True)
    merra_data.reset_index(drop=True, inplace=True)
    return merra_data


def load_all_neon_data(
        tool_weather_data_path=NEON_WEATHER_DATA_FOLDER,
        colname2varname=(
            ('priPrecipBulk',
             'rain_precip_mm_5min')),  # ('windSpeedMean','wind_speed_m_s')
        length=5,
        location='09',
        region='dalton',
        wind_height=2,  # 1,2,3 = 0.39, 1.88, 4.03
        local_time_zone='America/Anchorage'):

    path2search = f'*TOOL*/*{length}min*.csv'
    neon_files = glob.glob(f'{tool_weather_data_path}/{path2search}')
    if 'windSpeedMean' in colname2varname[0]:
        assert length in [2, 30
                         ], 'windSpeedMean is only available for 2 and 30 min'
        if wind_height == 1:
            neon_files = [f for f in neon_files if '.010.' in f]
        elif wind_height == 2:
            neon_files = [f for f in neon_files if '.020.' in f]
        elif wind_height == 3:
            neon_files = [f for f in neon_files if '.030.' in f]
        else:
            raise ValueError('wind_height should be 1,2,3')
    neon_df = []
    for file in neon_files:
        w_d = pd.read_csv(file)
        w_d['startDateTime'] = pd.to_datetime(
            w_d['startDateTime']).dt.tz_convert(local_time_zone).dt.tz_localize(
                None)
        w_d['endDateTime'] = pd.to_datetime(w_d['endDateTime']).dt.tz_convert(
            local_time_zone).dt.tz_localize(None)
        if location != '':
            w_d['location'] = location.lower()
        if region != '':
            w_d['region'] = region.lower()

        w_d['timestamp'] = w_d['startDateTime']
        w_d[colname2varname[1]] = w_d[colname2varname[0]]
        neon_df.append(w_d)

    # priPrecipFinalQF is the quality flag for precipitation
    # 0 is good, 1 is bad
    neon_df = pd.concat(neon_df)
    neon_df = neon_df.sort_values(by=['startDateTime'])
    neon_df = neon_df.reset_index(drop=True)

    return neon_df


def load_rows(csv_fname, region, short_locations, long_locations):
    with open(csv_fname, newline='', encoding='utf-8') as csvfile:
        csv_reader = list(csv.reader(csvfile))
        short = len(csv_reader[0]) == 14
        if region in short_locations and not short:
            raise Exception(f'short location has long csv {region}')
        if region in long_locations and short:
            raise Exception(f'long location has short csv {region}')

        return csv_reader, short


def get_random_rows(reader, file_per_location, station_years):
    # filter rows for available years
    rows_4_available_years = [
        row for row in reader if int(row[0]) in station_years
    ]
    rows_picked = random.choices(rows_4_available_years, k=file_per_location)
    return rows_picked


def parse_row(row, location, region, input_csv_headers):

    # Extract the year, month, day, and hour from the row
    year, month, day, hour = [int(row[x]) for x in range(4)]

    pd_row = {}

    # Add the location and region to the row
    pd_row['location'] = location.lower()
    pd_row['region'] = region.lower()

    # Compute the timestamp for the current offset
    timestamp = datetime.datetime(year, month, day, hour=hour)
    pd_row['timestamp'] = timestamp

    # Convert the data values to floats and add them to the row
    row[4:] = [float(x) for x in row[4:]]
    for label, data in zip(input_csv_headers, row[4:]):  # type: ignore
        pd_row[label] = data

    return pd_row


def parse_rows(rows_picked, location, region, short, short_headers,
               long_headers):

    # Determine which headers to use
    input_csv_headers = short_headers if short else long_headers

    # Use list comprehension to parse each row
    pd_rows = [
        parse_row(row, location, region, input_csv_headers)
        for row in rows_picked
    ]

    return pd_rows


def shift_row_timestamp_2_beginning_of_window(row, weather_data_freq):
    """Shift the timestamp of a row to the beginning of the 3-hour window."""
    timestamp = row['timestamp']
    timestamp = timestamp - datetime.timedelta(seconds=weather_data_freq)
    return row


def generate_extended_rows(pd_rows, timestamps_per_row, weather_data_freq):
    """Generate additional rows with equally
    spaced timestamps within the weather_data_freq window.

    if timestamps_per_row = 1, then timestamps are
         shifted to the middle of the window

    """

    if timestamps_per_row == 0:
        raise ValueError('timestamps_per_row must be greater than 0.')

    new_rows = []

    for row in pd_rows:
        timestamp = row['timestamp']

        timestamp_offsets = [
            datetime.timedelta(seconds=weather_data_freq * offset /
                               (timestamps_per_row + 1))
            for offset in range(1, timestamps_per_row + 1)
        ]

        for offset in timestamp_offsets:
            new_row = row.copy()
            new_timestamp = timestamp + offset
            new_row['timestamp'] = new_timestamp
            new_rows.append(new_row.copy())

    return new_rows


def csv_path_per_regloc(data_folder):

    station_csv = {}
    for region_path in glob.glob(f'{data_folder}/*'):
        locations = glob.glob(region_path + '/sm_products_by_station/*')
        region = Path(region_path).name.lower()
        for location_path in locations:
            location = Path(location_path).stem.split('_')[-1]
            if region != location[:-2].lower():
                print(region, location)
            location = location[-2:]
            # print(location_path)
            if region == 'ivvavik':
                location = 'sinp' + location
            station_csv[(region, location)] = location_path
            # print(region,location)
        # print(len(locations))
    return station_csv


def year_per_regloc(station_csv, file_properties_df):

    station_years = {}
    for region, location in station_csv.keys():
        region_filtered = file_properties_df[file_properties_df['region'] ==
                                             region]
        loc_reg_filtered = region_filtered[region_filtered['location'] ==
                                           location]

        # print(region,location)
        unique_years = (loc_reg_filtered.year.unique())
        unique_years = [int(year) for year in unique_years if int(year) > 2018]
        # print(unique_years)
        station_years[(region, location)] = unique_years
    return station_years


def shift_weather(adele_weather, shift_amount_hours):
    # shift_amount = -15
    adele_weather_shifted = adele_weather.copy()
    adele_weather_shifted['timestamp'] = adele_weather_shifted[
        'timestamp'] + pd.Timedelta(hours=shift_amount_hours)
    return adele_weather_shifted


def load_adele_weather_data(region,
                            location,
                            fname,
                            short_locations,
                            long_locations,
                            short_input_csv_headers,
                            long_input_csv_headers,
                            correct_timestamps=False):
    csv_reader, short = load_rows(fname, region, short_locations,
                                  long_locations)
    pd_rows = parse_rows(csv_reader, location, region, short,
                         short_input_csv_headers, long_input_csv_headers)
    data = pd.DataFrame(pd_rows)
    shift_amount2beginning = -(ADELE_WEATER_DATA_FREQ / 3600)
    data = shift_weather(data, shift_amount_hours=shift_amount2beginning)
    if correct_timestamps:
        data = shift_weather(data, shift_amount_hours=CORRECT_SHIFT_HOURS)
    else:
        raise ValueError('timestamps are not being Shifted to correction')
    data['rain_precip_mm'] = data['rain_precip'] * 1000
    data['rain_precip_mm_3hr'] = data['rain_precip_mm']
    return data


# Find the bin/row index for each timestamp in the list
def get_bin_index(interval_index, timestamp):
    try:
        index = interval_index.get_loc(timestamp)
    except KeyError:
        index = -1
    return index


def get_weather_index(weather_data,
                      weather_data_freq,
                      timestamp_fromat=TIMESTAMP_FORMAT):

    if not pd.api.types.is_datetime64_dtype(weather_data['timestamp']):
        weather_data['timestamp_start'] = weather_data['timestamp'].apply(
            lambda x: pd.to_datetime(x, format=timestamp_fromat))
    else:
        # Do nothing, the column is already in datetime type
        weather_data['timestamp_start'] = weather_data['timestamp']

    weather_data['timestamp_end'] = weather_data[
        'timestamp_start'] + pd.Timedelta(seconds=weather_data_freq)
    weather_data.sort_values(by='timestamp_start', inplace=True)

    interval_index = pd.IntervalIndex.from_arrays(
        weather_data['timestamp_start'],
        weather_data['timestamp_end'],
        closed='left')
    return interval_index


def get_random_timestamp(start, end):
    # Generate a random number of seconds between 0 and the duration of
    # the audio clip (in seconds)
    audio_duration = int((end - start).total_seconds())
    random_seconds = random.randint(0, audio_duration)

    # Add the random number of seconds to the start time and subtract 10 seconds
    # to get a random timestamp within the range of start and end with
    #  a buffer of 10 seconds
    random_timestamp = start + pd.Timedelta(
        seconds=random_seconds) - pd.Timedelta(seconds=10)

    # Ensure that the random timestamp is within the range of start and end
    random_timestamp = np.clip(random_timestamp, start, end)

    return random_timestamp


def check_time_diff(df, data_freq_seconds):
    # Calculate time difference between consecutive timestamps
    time_diff = df['timestamp'].diff()

    # Check if time difference is equal to MERRA_DATA_FREQ for first 10 rows
    if (time_diff.iloc[1:11] == pd.Timedelta(seconds=data_freq_seconds)).all():
        pass
    else:
        raise ValueError(
            f'Time difference is not equal to {data_freq_seconds}' +
            'for first 10 rows')


def add_weather2dataset(dataset,
                        weather_df,
                        weather_data_freq,
                        columns_to_transfer=None,
                        verbose=False):
    ''' Add weather data to the dataset

        Dataset is standard dataset with columns:
        ['region', 'location', 'Date', 'Start Time', 'End Time', 'Duration',

        weather_df is the weather data with columns:
        ['timestamp', 'region', 'location', *columns_to_transfer]

    '''
    weather_df = weather_df.copy()
    check_time_diff(weather_df, weather_data_freq)
    if columns_to_transfer is None:
        columns_to_transfer = ['rain_precip_mm_1hr']
    dataset_precip = {col: {} for col in columns_to_transfer}
    weather_interval_index = get_weather_index(
        weather_df, weather_data_freq=weather_data_freq)
    dataset = dataset.copy()
    for i, row in dataset.iterrows():
        region = row['region'].lower()
        location = row.get('Site ID', row.get('location', None)).lower()
        a_timestamp = pd.to_datetime(row['Date'] + ' ' + row['Start Time'])
        weather_bind_index = get_bin_index(weather_interval_index, a_timestamp)

        if weather_bind_index == -1:
            if verbose:
                print('no weather data for', a_timestamp, i)
            continue

        selected = (weather_df.iloc[weather_bind_index])
        selected = selected[selected['region'] == region]
        selected = selected[selected['location'] == location]
        if selected.shape[0] > 1:
            raise ValueError('more than one weather data for', location, region,
                             a_timestamp)
        if selected.empty:
            if verbose:
                print('no weather data for', location, region, a_timestamp)
            continue

        assert len(selected) == 1
        for col in columns_to_transfer:
            dataset_precip[col][row['Clip Path']] = selected[col].values[0]

    for col in columns_to_transfer:
        col_name = 'weather_timestamp' if col == 'timestamp' else col
        dataset[col_name] = dataset['Clip Path'].map(dataset_precip[col])

    return dataset

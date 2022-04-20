import pandas as pd
from datetime import datetime
#
# df = pd.read_csv('good_pagination.csv')
#
# df['TimeStamp'] = df['TimeStamp'].apply(lambda x: datetime.utcfromtimestamp(x / 1000).strftime('%Y-%m-%d'))
#
# # print(df['TimeStamp'].head())
#
# df.drop('Unnamed: 0', axis=1, inplace=True)
#
# df = df.sort_values(by='TimeStamp').reset_index(drop=True)
#
#
# for sensor_name in df['Sensor'].unique():
#     df[sensor_name] = df['Value'][df['Sensor'] == sensor_name]
#
# print(df.columns)
# # df = df.groupby(['TimeStamp'])[['pm25', 'pm1', 'pm10', 'co2', 'o3', 'cho2', 'no2', 'so2']].mean().reset_index()
# df = df.groupby(['TimeStamp'])[['pm25', 'pm1', 'pm10', 'co2', 'o3', 'cho2', 'no2', 'so2','LocationLat','LocationLong']].mean().reset_index()
#
#
# df.to_csv('last_step_pagination_110422.csv')
def average_per_days(df):
    # df = pd.read_csv('good_pagination.csv')

    df['TimeStamp'] = df['TimeStamp'].apply(lambda x: datetime.utcfromtimestamp(x / 1000).strftime('%Y-%m-%d'))

    # print(df['TimeStamp'].head())

    # df.drop('Unnamed: 0', axis=1, inplace=True)


    df = df.sort_values(by='TimeStamp').reset_index(drop=True)

    for sensor_name in df['Sensor'].unique():
        df[sensor_name] = df['Value'][df['Sensor'] == sensor_name]

    # print(df.columns)
    # df = df.groupby(['TimeStamp'])[['pm25', 'pm1', 'pm10', 'co2', 'o3', 'cho2', 'no2', 'so2']].mean().reset_index()
    df = df.groupby(['TimeStamp'])[
        ['pm25', 'pm1', 'pm10', 'co2', 'o3', 'cho2', 'no2', 'so2', 'LocationLat', 'LocationLong']].mean().reset_index()

    df.to_csv('last_step_pagination_110422.csv')
    return df

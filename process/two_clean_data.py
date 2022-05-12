# import os
#
# import pandas as pd
# from sklearn.impute import SimpleImputer
# #
# # df = pd.read_csv('last_step_pagination_110422.csv', infer_datetime_format=True)
# #
# # df = df.drop('Unnamed: 0', axis=1)
# #
# # df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], format="%Y-%m-%d %H:%M:%S")
# #
# # df.index = df['TimeStamp']
# # df.index.sort_values()
# # # print(df)
# # print(df.count())  # added sensor columns)
# #
# # # remove data before 2021
# # df = df[df.index > '2021-01-01']
# #
# # # add missing days
# # df = df.resample('D').mean()
# #
# #
# # # df['TimeStamp']=df.index
# #
# # # check how many rows are missing
# # def percentage(part, whole):
# #     return 100 * float(part) / float(whole)
# #
# #
# # row_count = df.shape[0]
# #
# # for c in df.columns:
# #     m_count = df[c].isna().sum()
# #
# #     if m_count > 0:
# #         print(f'{c} - {m_count} ({round(percentage(m_count, row_count), 3)}%) rows missing')
# #
# # df = df.drop(['LocationLat', 'LocationLong'], axis=1)
# #
# # print('before', df.describe())
# #
# # # add 0 instead of NaN for pm2.5, pm1 and pm10 as their min is around 0
# # zero_columns = ['pm25', 'pm1', 'pm10']
# # for column in zero_columns:
# #     df[column] = df[column].fillna(0)
# # print('columns', df.columns)
# #
# # sensor_column_names = ['pm25', 'pm1', 'pm10', 'co2', 'o3', 'cho2', 'no2', 'so2']
# # imp = SimpleImputer(strategy="most_frequent")
# # df_mean_imputed = pd.DataFrame(imp.fit_transform(df.iloc[:, 0:]), columns=sensor_column_names)
# # #
# # df_mean_imputed['TimeStamp'] = df.index
# # df_mean_imputed.index = df.index
# #
# # df_mean_imputed['TimeStamp'] = df.index
# # df_mean_imputed.index = df.index
# #
# # print('after', df_mean_imputed.describe())
# #
# # df_mean_imputed.to_csv('df_imputed_120422.csv')  # DONE NOW 120422
#
# def clean_data(df):
#     # df = pd.read_csv('last_step_pagination_110422.csv', infer_datetime_format=True)
#
#     # df = df.drop('Unnamed: 0', axis=1)
#
#     df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], format="%Y-%m-%d %H:%M:%S")
#
#     df.index = df['TimeStamp']
#     df.index.sort_values()
#     # print(df)
#     print(df.count())  # added sensor columns)
#
#     # remove data before 2021
#     df = df[df.index > '2021-01-01']
#
#     # add missing days
#     df = df.resample('D').mean()
#
#     # df['TimeStamp']=df.index
#
#     # check how many rows are missing
#     def percentage(part, whole):
#         return 100 * float(part) / float(whole)
#
#     row_count = df.shape[0]
#
#     for c in df.columns:
#         m_count = df[c].isna().sum()
#
#         if m_count > 0:
#             print(f'{c} - {m_count} ({round(percentage(m_count, row_count), 3)}%) rows missing')
#
#     df = df.drop(['LocationLat', 'LocationLong'], axis=1)
#
#     print('before', df.describe())
#
#     # add 0 instead of NaN for pm2.5, pm1 and pm10 as their min is around 0
#     # zero_columns = ['pm25', 'pm1', 'pm10']
#     # for column in zero_columns:
#     #     df[column] = df[column].fillna(0)
#     # print('columns', df.columns)
#     #todo check impute 0 or most frequent values for pm2.5, pm1 and pm10
#
#     sensor_column_names = ['pm25', 'pm1', 'pm10', 'co2', 'o3', 'cho2', 'no2', 'so2']
#     imp = SimpleImputer(strategy="most_frequent")
#     df_mean_imputed = pd.DataFrame(imp.fit_transform(df.iloc[:, 0:]), columns=sensor_column_names)
#     #
#     df_mean_imputed['TimeStamp'] = df.index
#     df_mean_imputed.index = df.index
#
#     df_mean_imputed['TimeStamp'] = df.index
#     df_mean_imputed.index = df.index
#
#     print('after', df_mean_imputed.describe())
#
#     # df_mean_imputed.to_csv('df_imputed_120422.csv')  # DONE NOW 120422
#     # df_mean_imputed.to_csv(os.path.join('process','df_imputed_120422.csv'))
#     return df_mean_imputed
import pandas as pd
from sklearn.impute import SimpleImputer
#
# df = pd.read_csv('last_step_pagination_110422.csv', infer_datetime_format=True)
#
# df = df.drop('Unnamed: 0', axis=1)
#
# df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], format="%Y-%m-%d %H:%M:%S")
#
# df.index = df['TimeStamp']
# df.index.sort_values()
# # print(df)
# print(df.count())  # added sensor columns)
#
# # remove data before 2021
# df = df[df.index > '2021-01-01']
#
# # add missing days
# df = df.resample('D').mean()
#
#
# # df['TimeStamp']=df.index
#
# # check how many rows are missing
# def percentage(part, whole):
#     return 100 * float(part) / float(whole)
#
#
# row_count = df.shape[0]
#
# for c in df.columns:
#     m_count = df[c].isna().sum()
#
#     if m_count > 0:
#         print(f'{c} - {m_count} ({round(percentage(m_count, row_count), 3)}%) rows missing')
#
# df = df.drop(['LocationLat', 'LocationLong'], axis=1)
#
# print('before', df.describe())
#
# # add 0 instead of NaN for pm2.5, pm1 and pm10 as their min is around 0
# zero_columns = ['pm25', 'pm1', 'pm10']
# for column in zero_columns:
#     df[column] = df[column].fillna(0)
# print('columns', df.columns)
#
# sensor_column_names = ['pm25', 'pm1', 'pm10', 'co2', 'o3', 'cho2', 'no2', 'so2']
# imp = SimpleImputer(strategy="most_frequent")
# df_mean_imputed = pd.DataFrame(imp.fit_transform(df.iloc[:, 0:]), columns=sensor_column_names)
# #
# df_mean_imputed['TimeStamp'] = df.index
# df_mean_imputed.index = df.index
#
# df_mean_imputed['TimeStamp'] = df.index
# df_mean_imputed.index = df.index
#
# print('after', df_mean_imputed.describe())
#
# df_mean_imputed.to_csv('df_imputed_120422.csv')  # DONE NOW 120422

def clean_data(df):
    # df = pd.read_csv('last_step_pagination_110422.csv', infer_datetime_format=True)

    df = df.drop('Unnamed: 0', axis=1)

    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], format="%Y-%m-%d %H:%M:%S")

    df.index = df['TimeStamp']
    df.index.sort_values()
    # print(df)
    print(df.count())  # added sensor columns)

    # remove data before 2021
    df = df[df.index > '2021-01-01']

    # add missing days
    df = df.resample('D').mean()

    # df['TimeStamp']=df.index

    # check how many rows are missing
    def percentage(part, whole):
        return 100 * float(part) / float(whole)

    row_count = df.shape[0]

    for c in df.columns:
        m_count = df[c].isna().sum()

        if m_count > 0:
            print(f'{c} - {m_count} ({round(percentage(m_count, row_count), 3)}%) rows missing')

    df = df.drop(['LocationLat', 'LocationLong'], axis=1)

    print('before', df.describe())

    # add 0 instead of NaN for pm2.5, pm1 and pm10 as their min is around 0
    zero_columns = ['pm25', 'pm1', 'pm10']
    for column in zero_columns:
        df[column] = df[column].fillna(0)
    print('columns', df.columns)

    sensor_column_names = ['pm25', 'pm1', 'pm10', 'co2', 'o3', 'cho2', 'no2', 'so2']
    imp = SimpleImputer(strategy="most_frequent")
    df_mean_imputed = pd.DataFrame(imp.fit_transform(df.iloc[:, 0:]), columns=sensor_column_names)
    #
    df_mean_imputed['TimeStamp'] = df.index
    df_mean_imputed.index = df.index

    df_mean_imputed['TimeStamp'] = df.index
    df_mean_imputed.index = df.index

    print('after', df_mean_imputed.describe())

    df_mean_imputed.to_csv('df_imputed_120422.csv')  # DONE NOW 120422

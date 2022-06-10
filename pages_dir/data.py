import streamlit as st
import pandas as pd
import os
from pages_dir import data_helper
from pages_dir import graphs
import numpy as np

from pages_dir.graphs import plot_heatmap

pd.set_option('precision', 2)

#
import requests
import pandas as pd
import zipfile

endpoint = "https://gitlab.com/api/v4/projects/35474313/jobs/artifacts/main/download?job=build-job"
# data = {"ip": "1.1.2.3"}
headers = {"Authorization": "Bearer glpat-3jZqLV1_yc-D7bX-v-5z"}


# test check
@st.cache(ttl=60 * 60 * 24)
def get_gitlab_data():
    """
    This function return a pandas DataFrame with the data from gitlab.
    """
    r = requests.get(endpoint, headers=headers)

    zip_file_name = 'test_downloaded.zip'
    open(zip_file_name, 'wb').write(r.content)

    with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
        zip_ref.extractall(os.path.join('process', "test_new"))

    gitlab_df = pd.read_csv(os.path.join('process', 'test_new', 'test.csv'))
    return gitlab_df


# gitlab_df = get_gitlab_data()


#

@st.cache
def get_raw_data():
    """
    This function return a pandas DataFrame with the raw data.
    """
    raw_df = pd.read_csv(os.path.join('process', 'last_step_pagination_110422.csv'))
    # raw_df = pd.read_csv(os.path.join('process', 'final_pagination_110422.csv'))
    # raw_df = raw_df.drop(['Unnamed: 0'], axis=1)
    return raw_df


# @st.cache
# @st.cache(allow_output_mutation=True)
@st.cache
def get_clean_data():
    """
    This function return a pandas DataFrame with the clean data.
    """
    # clean_df = pd.read_csv(os.path.join('process', 'df_imputed_120422.csv'))
    # clean_df = pd.read_csv(os.path.join('process', 'df_imputed_120422.csv'))
    # clean_df = clean_df.drop(['TimeStamp.1'], axis=1)
    clean_df = pd.read_csv(os.path.join('process', 'last_step_pagination_110422.csv'), infer_datetime_format=True)

    # df = df.drop('Unnamed: 0', axis=1)

    clean_df['TimeStamp'] = pd.to_datetime(clean_df['TimeStamp'], format="%Y-%m-%d %H:%M:%S")

    # clean_df.index = clean_df['TimeStamp']
    # clean_df.index.sort_values()

    # remove data before 2021
    clean_df = clean_df[clean_df['TimeStamp'] > '2021-01-01']

    # add missing days
    # df = df.resample('5h', on='TimeStamp').mean()
    clean_df = clean_df.resample('3h', on='TimeStamp').mean()

    #
    # impute missing data with simple imputer
    sensor_column_names = ['pm25', 'pm1', 'pm10', 'co2', 'o3', 'cho2', 'no2', 'so2']
    # imp = SimpleImputer(strategy="most_frequent")
    # df_mean_imputed = pd.DataFrame(imp.fit_transform(df.iloc[:, 0:]), columns=sensor_column_names)

    # clean_df = clean_df.interpolate(method='linear', axis=0).ffill().bfill()
    clean_df = clean_df.interpolate(method='linear', axis=0).ffill().bfill()

    clean_df['TimeStamp'] = clean_df.index
    # clean_df.index = clean_df.index
    print("CLEAN DF ", clean_df.columns)
    #
    # df_mean_imputed['TimeStamp'] = df.index
    # df_mean_imputed.index = df.index

    return clean_df


@st.cache(allow_output_mutation=True)
def get_clean_data_fb():
    """
    This function return a pandas DataFrame with the clean data.
    """
    # clean_df = pd.read_csv(os.path.join('process', 'df_imputed_120422.csv'))
    # clean_df = pd.read_csv(os.path.join('process', 'df_imputed_120422.csv'))
    # clean_df = clean_df.drop(['TimeStamp.1'], axis=1)
    clean_df = pd.read_csv(os.path.join('process', 'last_step_pagination_110422.csv'), infer_datetime_format=True)

    # df = df.drop('Unnamed: 0', axis=1)

    clean_df['TimeStamp'] = pd.to_datetime(clean_df['TimeStamp'], format="%Y-%m-%d %H:%M:%S")

    # clean_df.index = clean_df['TimeStamp']
    # clean_df.index.sort_values()

    # remove data before 2021
    clean_df = clean_df[clean_df['TimeStamp'] > '2021-01-01']

    # add missing days
    # df = df.resample('5h', on='TimeStamp').mean()
    clean_df = clean_df.resample('24h', on='TimeStamp').mean()

    #
    # impute missing data with simple imputer
    sensor_column_names = ['pm25', 'pm1', 'pm10', 'co2', 'o3', 'cho2', 'no2', 'so2']
    # imp = SimpleImputer(strategy="most_frequent")
    # df_mean_imputed = pd.DataFrame(imp.fit_transform(df.iloc[:, 0:]), columns=sensor_column_names)

    clean_df = clean_df.interpolate(method='linear', axis=0).ffill().bfill()

    clean_df['TimeStamp'] = clean_df.index
    # clean_df.index = clean_df.index
    print("CLEAN DF ", clean_df.columns)
    #
    # df_mean_imputed['TimeStamp'] = df.index
    # df_mean_imputed.index = df.index

    return clean_df


raw_df = get_raw_data()
clean_df = get_clean_data()
clean_df_fb = get_clean_data_fb()


def create_visualization(data):
    """histogram and line plot"""
    sensor_names = list(data.columns)
    try:
        for item in ['TimeStamp', 'LocationLat', 'LocationLong','Source','Measurement']:
            sensor_names.remove(item)
    except ValueError:
        pass
    #st.header('Histogram Visualization')

    select_sensor = st.selectbox(
        'Select a sensor from the list',
        [i for i in sensor_names]
    )

    st.subheader('Histogram')
    # sensor_names = list(data.columns)
    # for item in ['TimeStamp', 'LocationLat', 'LocationLong']:
    #     sensor_names.remove(item)

    fig = graphs.plot_histogram(data=data, x=select_sensor, height=500, width=950)
    st.plotly_chart(fig)
    
    st.subheader('Line Plot')

    fig = graphs.plot_line(data=data, x=data['TimeStamp'], y=select_sensor, height=500, width=700)

    st.plotly_chart(fig)


def app():
    st.title('Data')

    # st.write("This is the `Data` page of the multi-page app.")
    #
    # st.dataframe(gitlab_df)
    #
    st.write("The following is the `air pollution` dataset.")

    type_of_data = st.radio(
        "Type of Data",
        ('Raw Data', 'Clean Data'),
        help='Data source that will be used in the analysis'
    )

    if type_of_data == 'Raw Data':
        data = raw_df.copy()
        st.dataframe(data)
        # st.dataframe(data.style.format({"E": "{:.2f}"}))

    elif type_of_data == 'Clean Data':
        data = clean_df.copy()
        st.dataframe(data)
        # st.dataframe(data.style.format({"E": "{:.2f}"}))
    else:
        data = raw_df.copy()

    # todo add clean data
    with st.container():
        st.header('Descriptive Statistics\n')

        descriptive_df = data_helper.summary_table(data)
        st.dataframe(descriptive_df.style.format({"E": "{:.2f}"}))

    with st.expander("See further description of data"):
        st.dataframe(data.describe())

    create_visualization(data)
    # histogram_visualization(data)

    st.subheader('Correlation Matrix')

    corr_matrix = data[['pm25', 'pm10', 'pm1', 'co2', 'o3', 'cho2', 'no2', 'so2']].corr()

    fig = plot_heatmap(corr_matrix)

    # fig = graphs.plot_heatmap(corr_matrix=corr_matrix, height=500, margin=750)
    # #
    # print('.tolist()', np.round(corr_matrix.values, 2))
    st.plotly_chart(fig)

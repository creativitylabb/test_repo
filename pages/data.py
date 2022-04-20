import streamlit as st
import pandas as pd
import os
from pages import data_helper
from pages import graphs
import numpy as np

from pages.graphs import plot_heatmap

pd.set_option('precision', 2)


@st.cache
def get_raw_data():
    """
    This function return a pandas DataFrame with the raw data.
    """
    raw_df = pd.read_csv(os.path.join('process', 'last_step_pagination_110422.csv'))
    # raw_df = pd.read_csv(os.path.join('process', 'final_pagination_110422.csv'))
    raw_df = raw_df.drop(['Unnamed: 0'], axis=1)
    return raw_df


# @st.cache
@st.cache(allow_output_mutation=True)
def get_clean_data():
    """
    This function return a pandas DataFrame with the clean data.
    """
    clean_df = pd.read_csv(os.path.join('process', 'df_imputed_120422.csv'))
    clean_df = clean_df.drop(['TimeStamp.1'], axis=1)

    return clean_df


raw_df = get_raw_data()
clean_df = get_clean_data()


def create_visualization(data):
    """histogram and line plot"""
    sensor_names = list(data.columns)
    try:
        for item in ['TimeStamp', 'LocationLat', 'LocationLong']:
            sensor_names.remove(item)
    except ValueError:
        pass
    st.header('Histogram Visualization')

    select_sensor = st.selectbox(
        'Select a sensor from the list',
        [i for i in sensor_names]
    )

    st.subheader('Scatterplot')
    # sensor_names = list(data.columns)
    # for item in ['TimeStamp', 'LocationLat', 'LocationLong']:
    #     sensor_names.remove(item)

    fig = graphs.plot_histogram(data=data, x=select_sensor, height=500, width=950)
    st.plotly_chart(fig)

    fig = graphs.plot_line(data=data, x=data['TimeStamp'], y=select_sensor, height=500, width=700)

    st.plotly_chart(fig)


def app():
    st.title('Data')

    # st.write("This is the `Data` page of the multi-page app.")

    st.write("The following is the `air pollution` dataset.")

    type_of_data = st.radio(
        "Type of Data",
        ('Raw Data', 'Clean Data'),
        help='Data source that will be used in the analysis'
    )

    if type_of_data == 'Raw Data':
        data = raw_df.copy()
        st.dataframe(data.style.format({"E": "{:.2f}"}))

    if type_of_data=='Clean Data':
        data = clean_df.copy()
        st.dataframe(data.style.format({"E": "{:.2f}"}))

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

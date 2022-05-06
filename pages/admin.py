import streamlit as st
import os.path

from pages.models.fb_create_models import create_models
from pages.models.lstm_create_models import create_model_lstm
from process.one_pagination import scroll_page, rename_columns
from process.two_clean_data import clean_data
from process.three_create_average_per_days import average_per_days
import pandas as pd
import streamlit_authenticator as stauth


# test check
@st.cache(ttl=60 * 60 * 24)
def get_pagination_data():
    # df = pd.DataFrame()
    #
    # pagination_df = scroll_page('brasov-dev', '1m', df)
    # pagination_df = rename_columns(pagination_df)
    # pagination_df = average_per_days(pagination_df)
    # pagination_df.to_csv(os.path.join('process', 'last_step_pagination_110422.csv'))
    pagination_df=pd.read_csv(os.path.join('process', 'last_step_pagination_110422.csv'))
    pagination_df = pagination_df.drop('Unnamed: 0', axis=1)

    clean_data_df = clean_data(pagination_df)
    clean_data_df.to_csv(os.path.join('process', 'df_imputed_120422.csv'))

    return clean_data_df


pagination_data = get_pagination_data()

# read df for lstm and fb prophet
df = pd.read_csv(os.path.join('process', 'df_imputed_120422.csv'), infer_datetime_format=True)


@st.cache(ttl=60 * 60 * 25)
def run_fb_model(df):
    # df = pd.read_csv(os.path.join('process', 'df_imputed_120422.csv'), infer_datetime_format=True)

    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], format="%Y-%m-%d %H:%M:%S")

    df.index = df['TimeStamp']
    df.index.sort_values()

    create_models(df, 'pm25')
    create_models(df, 'pm1')
    create_models(df, 'pm10')


@st.cache(ttl=60 * 60 * 25)
def run_lstm_model(df):
    # df = pd.read_csv(os.path.join('process', 'df_imputed_120422.csv'), infer_datetime_format=True)

    df.index = df['TimeStamp']
    df = df.drop('TimeStamp', axis=1)

    df = df[['pm25', 'pm1', 'pm10']]

    create_model_lstm(df, column_index=0, sensor_name='pm25')  # pm25
    # create_model_lstm(df, column_index=1, sensor_name='pm1')  # pm1
    # create_model_lstm(df, column_index=2, sensor_name='pm10')  # pm10


# run_model_fb = run_fb_model(df)
# run_model_lstm = run_lstm_model(df)


def app():
    #     """Admin page"""
    st.title('Admin')
    hashed_passwords = stauth.Hasher(st.secrets["db_credentials"]["passwords"]).generate()

    # st.write("My cool secrets:", st.secrets["db_credentials"]["usernames"])
    # st.write("My cool secrets:", st.secrets["db_credentials"]["passwords"])

    authenticator = stauth.Authenticate(st.secrets["db_credentials"]["names"],
                                        st.secrets["db_credentials"]["usernames"], hashed_passwords,
                                        'some_cookie_name', 'some_signature_key', cookie_expiry_days=30)
    name, authentication_status, username = authenticator.login('Login', 'main')

    if authentication_status:
        st.write('Welcome *%s*' % (name))

        authenticator.logout('Logout', 'main')

        st.write(pagination_data)

        if st.sidebar.button('Download pagination data'):
            st.write('downloading started...')
            get_pagination_data()

        if st.sidebar.button('Run Facebook Prophet models'):
            st.write('Run Facebook Prophet started...')
            run_fb_model(df)

        if st.sidebar.button('Run LSTM models'):
            st.write('LSTM started...')
            run_lstm_model(df)

        # st.write(pagination_data)
    elif not authentication_status:
        st.error('Username/password is incorrect')
    elif authentication_status is None:
        st.warning('Please enter your username and password')
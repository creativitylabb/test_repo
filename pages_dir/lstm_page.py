import streamlit as st
from pages_dir.data import clean_df
import os
import pandas as pd

from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import load_model

from pages_dir.helper_plot import plot_fb_data
from pages_dir.home import create_metrics
from plotly import graph_objs as go
import numpy as np
from scipy.stats.stats import pearsonr
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

df_clean = clean_df.copy()


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    #     return (np.sum(np.abs((y_pred-y_true) / y_true))/len(y_true))*100
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def cv_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return (np.sqrt(np.sum(np.square(y_pred - y_true)) / (len(y_true) - 1)) / np.mean(y_true)) * 100


def smape(A, F):
    return 100 / len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))


def model_lstm(selected_sensor='pm25'):
    """predict all data"""
    # todo choose sensor

    if st.sidebar.button('Predict'):
        with st.spinner('Loading the predictions...'):
            # model = pickle.load(
            #     open(os.path.join('pages', 'models', 'lstm_model_pm25.pkl'), 'rb'))  # predicts pm25

            # todo change name
            model = load_model(os.path.join('pages_dir', 'models', 'lstm_model_' + str(selected_sensor) + '.h5'))

            df = df_clean.copy()
            # df.index = df['TimeStamp']

            df = df[['pm25', 'pm1', 'pm10']]

            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(df)

            features = data_scaled  # pm1	pm10
            if selected_sensor == 'pm25':
                target = data_scaled[:, 0]  # pm25 #todo change!
            elif selected_sensor == 'pm1':
                target = data_scaled[:, 1]  # pm1 #todo change!
            elif selected_sensor == 'pm10':
                target = data_scaled[:, 2]  # pm10 #todo change!

            x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=123,
                                                                shuffle=False)  # just pass features, not time
            win_length = 1  # random
            batch_size = 24  # 32  #training to be faster
            test_generator = TimeseriesGenerator(x_test, y_test, length=win_length, sampling_rate=1,
                                                 batch_size=batch_size)

            predictions = model.predict_generator(test_generator)

            df_pred = pd.concat([pd.DataFrame(predictions), pd.DataFrame(x_test[:, 1:][win_length:])], axis=1)
            df_pred.head()  # df with scaled values

            rev_trans = scaler.inverse_transform(df_pred)

            df_final = df[predictions.shape[0] * -1:]

            # df_final['pm25_Pred'] = rev_trans[:, 0]  # get only first column, pm10 predicted column
            df_final['predicted ' + str(selected_sensor)] = rev_trans[:,
                                                            0]  # get only first column, pm10 predicted column
            print('df_final ', df_final)
            # df_final[['pm25', 'pm25_Pred']].plot()
            df_final['TimeStamp'] = df_final.index
            #

            #

            print("DF FINAL ", df_final)

            plot_fb_data(df_final, selected_sensor, 'predicted ' + str(selected_sensor), 'TimeStamp')

            df_final['predicted-actual'] = df_final['predicted ' + str(selected_sensor)] - df_final[selected_sensor]

            with st.expander("See metrics"):
                st.dataframe(df_final[['predicted-actual', 'predicted ' + str(selected_sensor), selected_sensor]])
                create_metrics(df_final, selected_sensor)
                #

                mape_score_test = round(mean_absolute_percentage_error(df_final[selected_sensor],
                                                                       df_final['predicted ' + str(selected_sensor)]),
                                        2)

                cv_score_test = round(
                    cv_error(df_final[selected_sensor], df_final['predicted ' + str(selected_sensor)]), 2)

                smape_score_test = round(
                    smape(df_final[selected_sensor], df_final['predicted ' + str(selected_sensor)]), 2)

                pearson_score_test = round(
                    pearsonr(df_final['predicted ' + str(selected_sensor)], df_final[selected_sensor])[0], 3)

                st.write("mape_score_test ", mape_score_test)
                st.write("cv_score_test ", cv_score_test)
                st.write("smape_score_test ", smape_score_test)
                st.write("pearson_score_test ", pearson_score_test)


#


def future_data(selected_sensor):
    """predict data for 5 next days"""
    if st.sidebar.button('Predict'):
        with st.spinner('Loading the predictions...'):
            # model = pickle.load(
            #     open(os.path.join('pages', 'models', 'lstm_model_pm25.pkl'), 'rb'))  # predicts pm25

            # todo change name
            model = load_model(os.path.join('pages_dir', 'models', 'lstm_model_' + str(selected_sensor) + '.h5'))

            # if selected_sensor == 'pm25':
            #     model = load_model(os.path.join('pages', 'models', 'lstm_model_pm25.h5'))
            # elif selected_sensor == 'pm1':
            #     model = load_model(os.path.join('pages', 'models', 'lstm_model_pm1.h5'))
            # elif selected_sensor == 'pm10':
            #     model = load_model(os.path.join('pages', 'models', 'lstm_model_pm10.h5'))
            df = df_clean.copy()
            df.index = df['TimeStamp']

            #
            last_date = df['TimeStamp'].max()

            dates = pd.date_range(last_date, periods=6)
            dates = dates[dates > last_date]  # Drop start if equals last_date

            df2 = dates.to_frame(index=False, name='TimeStamp')

            # df2['ds'] = df2['TimeStamp']
            # df2['y'] = ''  # df['pm25']
            # print('df2 ',df2)
            df2 = pd.concat([df, df2])
            df2 = df2.reset_index(drop=True)
            df2.index = df2['TimeStamp']
            print('df22 ', df2)
            df2['pm25'][-5:] = df['pm25'][-5:]  # 0
            df2['pm1'][-5:] = df['pm1'][-5:]  # 0
            df2['pm10'][-5:] = df['pm10'][-5:]  # 0

            df = df2
            #

            df = df[['pm25', 'pm1', 'pm10']]
            # df = df.drop('TimeStamp', axis=1)

            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(df)

            features = data_scaled  # pm1	pm10
            if selected_sensor == 'pm25':
                target = data_scaled[:, 0]  # pm25 #todo change!
            elif selected_sensor == 'pm1':
                target = data_scaled[:, 1]  # pm1 #todo change!
            elif selected_sensor == 'pm10':
                target = data_scaled[:, 2]  # pm10 #todo change!

            x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=123,
                                                                shuffle=False)  # just pass features, not time
            win_length = 1  # random
            batch_size = 24  # 32  #training to be faster
            test_generator = TimeseriesGenerator(x_test, y_test, length=win_length, sampling_rate=1,
                                                 batch_size=batch_size)

            predictions = model.predict_generator(test_generator)

            df_pred = pd.concat([pd.DataFrame(predictions), pd.DataFrame(x_test[:, 1:][win_length:])], axis=1)
            df_pred.head()  # df with scaled values

            rev_trans = scaler.inverse_transform(df_pred)

            df_final = df[predictions.shape[0] * -1:]

            # df_final['pm25_Pred'] = rev_trans[:, 0]  # get only first column, pm10 predicted column
            df_final['predicted ' + str(selected_sensor)] = rev_trans[:,
                                                            0]  # get only first column, pm10 predicted column
            print('df_final ', df_final)
            # df_final[['pm25', 'pm25_Pred']].plot()
            df_final['TimeStamp'] = df_final.index

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_final['TimeStamp'][-5:], y=df_final['predicted ' + str(selected_sensor)][-5:],
                                     name='Predicted ' + str(selected_sensor)))
            fig.layout.update(title_text='Predicted Data', xaxis_rangeslider_visible=False)
            st.plotly_chart(fig)


def app():
    """homepage of the app: plot and table"""
    st.title('LSTM')
    selected_sensor_lstm = st.sidebar.selectbox(
        'Choose sensor:',
        ('pm25', 'pm1', 'pm10')
    )
    st.sidebar.write('Chosen sensor: ', selected_sensor_lstm)
    lstm_opt = st.sidebar.selectbox(
        'What data you want to predict?',
        ('All', '5 days in the future'))
    if lstm_opt == 'All':
        model_lstm(selected_sensor=selected_sensor_lstm)
    if lstm_opt == '5 days in the future':
        future_data(selected_sensor=selected_sensor_lstm)

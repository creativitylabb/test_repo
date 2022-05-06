import pandas as pd
import os
from fbprophet import Prophet
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import TimeseriesGenerator


#
# def create_models_lstm(sensor_name):
#     options = ['pm25', 'pm1', 'pm10']
#     options.remove(sensor_name)
#     # use pm1, pm10 to predict pm25
#     df_final = aq_df[['TimeStamp', 'pm25', 'pm1', 'pm10']].rename({'TimeStamp': 'ds', sensor_name: 'y'}, axis='columns')
#
#     print(df_final.head())
#
#     # df_final.set_index('ds')[['y','pm1','pm10']].plot()#pm25,pm1,pm10
#
#     eighty_percent = int(80 / 100 * len(df_final))
#
#     train_df = df_final[:eighty_percent]
#     # train_df.shape
#     test_df = df_final[eighty_percent:]
#
#     model = Prophet(interval_width=0.9)
#     model.add_regressor(options[0], standardize=False)
#     model.add_regressor(options[1], standardize=False)
#     model.fit(train_df)
#
#     pickle.dump(model, open('fb_prophet_model_' + str(sensor_name) + '.pkl', 'wb'))
#     # pickle.dump(model,open('fb_prophet_model_pm1.pkl','wb'))
#     # pickle.dump(model,open('fb_prophet_model_pm10.pkl','wb'))
#

# cur_path = os.path.dirname(__file__)

# print(cur_path)
# new_path = os.path.join('pages', 'resources', 'df_imputed_120422.csv')

# new_path = os.path.relpath('../../process/df_imputed_120422.csv', cur_path)
#
# df = pd.read_csv(new_path, infer_datetime_format=True)
#
# df.index = df['TimeStamp']
# df = df.drop('TimeStamp', axis=1)
#
# df = df[['pm25', 'pm1', 'pm10']]
# print(
#     df
# )


def create_model_lstm(df, column_index, sensor_name):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)

    features = data_scaled  # pm1	pm10
    target = data_scaled[:, column_index]  # pm25
    # target = data_scaled[:, 0]  # pm25

    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.1, random_state=123,
                                                        shuffle=False)  # just pass features, not time
    df.index = pd.to_datetime(df.index)

    win_length = 2 #20 is worse r=0.13  # random
    batch_size = 128  # 32 r=0.31  #training to be faster
    # num_features=3

    num_features = len(df.columns)  # 8

    train_generator = TimeseriesGenerator(x_train, y_train, length=win_length, sampling_rate=1, batch_size=batch_size)
    test_generator = TimeseriesGenerator(x_test, y_test, length=win_length, sampling_rate=1, batch_size=batch_size)

    ################################
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(150, input_shape=(win_length, num_features),
                                   return_sequences=True))  # take every obs into account

    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))

    model.add(tf.keras.layers.LSTM(50, return_sequences=True))
    # model.add(tf.keras.layers.LeakyReLU(alpha=0.5))  # activation function
    model.add(tf.keras.layers.Dropout(0.2))  # make sure not overfit
    model.add(tf.keras.layers.LSTM(60, return_sequences=False))  # return only one hidden state
    # model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(1))

    print(model.summary())

    # early_stopping=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=2,mode='min')
    # model.compile(loss=tf.losses.MeanSquaredError(),optimizer=tf.optimizers.Adam(),metrics=[tf.metrics.MeanAbsoluteError()])
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=["accuracy"])

    history = model.fit(train_generator, epochs=650, validation_data=test_generator, shuffle=False)
    # history=model.fit(train_generator,epochs=50,validation_data=test_generator,shuffle=False,callbacks=[early_stopping])

    model.evaluate_generator(test_generator, verbose=0)  # evaluate model with test data

    # predictions = model.predict_generator(test_generator)
    #
    # df_pred = pd.concat([pd.DataFrame(predictions), pd.DataFrame(x_test[:, 1:][win_length:])], axis=1)
    # df_pred.head()  # df with scaled values
    #
    # rev_trans = scaler.inverse_transform(df_pred)
    #
    # df_final = df[predictions.shape[0] * -1:]
    #
    #
    # df_final['pm25_Pred'] = rev_trans[:, 0]  # get only first column, pm10 predicted column
    # df_final[['pm25', 'pm25_Pred']].plot()
    model.save(
        os.path.join('pages', 'models', 'lstm_model_' + str(sensor_name) + '.h5'))  # creates a HDF5 file 'my_model.h5'

# create_model_lstm(column_index=0, sensor_name='pm25')  # pm25
# create_model_lstm(column_index=1, sensor_name='pm1')  # pm1
# create_model_lstm(column_index=2, sensor_name='pm10')  # pm10


# good 0.32
#     model = tf.keras.Sequential()
#     model.add(tf.keras.layers.LSTM(150, input_shape=(win_length, num_features),
#                                    return_sequences=True))  # take every obs into account
#
#     model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
#
#     model.add(tf.keras.layers.LSTM(50, return_sequences=True))
#     # model.add(tf.keras.layers.LeakyReLU(alpha=0.5))  # activation function
#     model.add(tf.keras.layers.Dropout(0.2))  # make sure not overfit
#     model.add(tf.keras.layers.LSTM(60, return_sequences=False))  # return only one hidden state
#     # model.add(tf.keras.layers.Dropout(0.5))
#
#     model.add(tf.keras.layers.Dense(1))
#
#     print(model.summary())
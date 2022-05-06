import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import load_model
from matplotlib import pyplot as plt

#good is df_imputed_120422
df = pd.read_csv('df_imputed_120422.csv', infer_datetime_format=True)
# df = pd.read_csv('df_mean_imputed_current.csv', infer_datetime_format=True)

df.index = df['TimeStamp']
df = df.drop('TimeStamp', axis=1)

# df = df[['pm25', 'pm1', 'pm10']]  #predict pm2.5 using pm2.5 and pm1 and pm10
df = df[['pm25', 'pm10']] #predict pm2.5 using pm2.5 and pm10

def create_model_lstm(df, column_index, sensor_name):
    # scaler = MinMaxScaler()
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)

    features = data_scaled  # pm1	pm10

    target = data_scaled[:, column_index]  # pm25
#
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=123,
                                                        shuffle=False)  # just pass features, not time
    df.index = pd.to_datetime(df.index)

    win_length = 20 #20 is worse r=0.13  # random
    batch_size = 128  # 32 r=0.31  #training to be faster
    # num_features=3

    num_features = len(df.columns)  # 8

    train_generator = TimeseriesGenerator(x_train, y_train, length=win_length, sampling_rate=1, batch_size=batch_size)
    test_generator = TimeseriesGenerator(x_test, y_test, length=win_length, sampling_rate=1, batch_size=batch_size)

#     ################################
#     model = tf.keras.Sequential()
    # model.add(tf.keras.layers.LSTM(150, input_shape=(win_length, num_features),
    #                                return_sequences=True))  # take every obs into account
    # model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    # model.add(tf.keras.layers.LSTM(50, return_sequences=True))
    # model.add(tf.keras.layers.LeakyReLU(alpha=0.5))  # activation function
    # model.add(tf.keras.layers.Dropout(0.3))  # make sure not overfit
    # model.add(tf.keras.layers.LSTM(20, return_sequences=False))  # return only one hidden state
    # model.add(tf.keras.layers.Dropout(0.5))

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(150, input_shape=(win_length, num_features),
                                   return_sequences=True))  # take every obs into account
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(tf.keras.layers.LSTM(50, return_sequences=True))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))  # activation function
    model.add(tf.keras.layers.Dropout(0.3))  # make sure not overfit
    model.add(tf.keras.layers.LSTM(20, return_sequences=False))  # return only one hidden state
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(1, activation='sigmoid')) #to check
    print(model.summary())

    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=["accuracy"])
    history = model.fit(train_generator, epochs=650, validation_data=test_generator, shuffle=False)
    # model.evaluate_generator(test_generator, verbose=0)  # evaluate model with test data
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_acc', 'val_acc'], loc='upper left')
    plt.show()

#
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_losss', 'validation_loss'], loc='upper right')
    plt.show()
#
    model.save(
        'lstm_model_' + str(sensor_name) + '.h5')  # creates a HDF5 file 'my_model.h5'

create_model_lstm(df,column_index=0, sensor_name='pm25')  # pm25
# # create_model_lstm(column_index=1, sensor_name='pm1')  # pm1
# # create_model_lstm(column_index=2, sensor_name='pm10')  # pm10
#
# selected_sensor='pm25'
#
#
#
#
#
# # todo change name
# model = load_model(os.path.join('pages', 'models', 'lstm_model_' + str(selected_sensor) + '.h5'))
#
# df.index = df['TimeStamp']
#
# df = df[['pm25', 'pm1', 'pm10']]
# # df = df.drop('TimeStamp', axis=1)
#
# scaler = MinMaxScaler()
# data_scaled = scaler.fit_transform(df)
#
# features = data_scaled  # pm1	pm10
# if selected_sensor == 'pm25':
#     target = data_scaled[:, 0]  # pm25 #todo change!
# elif selected_sensor == 'pm1':
#     target = data_scaled[:, 1]  # pm1 #todo change!
# elif selected_sensor == 'pm10':
#     target = data_scaled[:, 2]  # pm10 #todo change!
#
# x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=123,
#                                                     shuffle=False)  # just pass features, not time
# win_length = 2  # random
# batch_size = 128  # 32  #training to be faster
# test_generator = TimeseriesGenerator(x_test, y_test, length=win_length, sampling_rate=1,
#                                      batch_size=batch_size)
#
# predictions = model.predict_generator(test_generator)
#
# df_pred = pd.concat([pd.DataFrame(predictions), pd.DataFrame(x_test[:, 1:][win_length:])], axis=1)
# df_pred.head()  # df with scaled values
#
# rev_trans = scaler.inverse_transform(df_pred)
#
# df_final = df[predictions.shape[0] * -1:]
#
# df_final['predicted ' + str(selected_sensor)] = rev_trans[:,
#                                                 0]  # get only first column, pm10 predicted column
# print('df_final ', df_final)
# df_final['TimeStamp'] = df_final.index
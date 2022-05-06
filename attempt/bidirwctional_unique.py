# import streamlit as st
import pandas as pd
import numpy as np
# import pydeck as pdk
import tensorflow as tf
# import plotly.express as px
import matplotlib.pyplot as plt

dataframe = pd.read_csv('df_imputed_120422.csv',infer_datetime_format=True)

series = np.asarray(dataframe.columns, dtype=float)

# time = np.arange(4 * 365 + 1, dtype="float32")
# time = np.arrange(dataframe['TimeStamp'])
baseline = 10
split_time = 400
time_train = dataframe['TimeStamp'][:split_time]
x_train = series[:split_time]
time_valid = dataframe['TimeStamp'][split_time:]
x_valid = series[split_time:]

window_size = 20
batch_size = 32
shuffle_buffer_size = 1000


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


def plot_series(time, series, format="-", start=0, end=None):
    plt.figure(figsize=(10, 6))
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()


tf.keras.backend.clear_session()
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

model = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                           input_shape=[None]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 100.0)
])

model.compile(loss="mae", optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9))
history = model.fit(dataset, epochs=50)

forecast = []
results = []
for time in range(len(series) - window_size):
    forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

forecast = forecast[split_time - window_size:]
results = np.array(forecast)[:, 0, 0]

# f1 = plot_series(time_valid, results)
# st.pyplot(f1)
#
# p = tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()
# st.markdown("## Mean Absolute Error: %s" % (p))
#
# st.write("Review the Results")

# mae=history.history['mae']
loss = history.history['loss']

epochs = range(len(loss))


def plot2():
    # plt.plot(epochs, mae, 'r')
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, 'b')
    plt.title('Training Loss')
    plt.xlabel("Epochs")
    plt.ylabel("MAE")
    plt.legend(["Loss"])
    plt.grid(True)
    plt.show()


# q = plot2()
# st.pyplot(q)
plot2()

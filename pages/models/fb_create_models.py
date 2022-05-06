import pandas as pd
import os
from fbprophet import Prophet
import pickle


def create_models(aq_df,sensor_name):
    options = ['pm25', 'pm1', 'pm10']
    options.remove(sensor_name)
    # use pm1, pm10 to predict pm25
    df_final = aq_df[['TimeStamp', 'pm25', 'pm1', 'pm10']].rename({'TimeStamp': 'ds', sensor_name: 'y'}, axis='columns')

    print(df_final.head())

    # df_final.set_index('ds')[['y','pm1','pm10']].plot()#pm25,pm1,pm10

    eighty_percent = int(80 / 100 * len(df_final))

    train_df = df_final[:eighty_percent]
    # train_df.shape
    test_df = df_final[eighty_percent:]

    model = Prophet(interval_width=0.9)
    model.add_regressor(options[0], standardize=False)
    model.add_regressor(options[1], standardize=False)
    model.fit(train_df)

    pickle.dump(model, open(os.path.join('pages','models','fb_prophet_model_' + str(sensor_name) + '.pkl'), 'wb'))
    # pickle.dump(model,open('fb_prophet_model_pm1.pkl','wb'))
    # pickle.dump(model,open('fb_prophet_model_pm10.pkl','wb'))


# cur_path = os.path.dirname(__file__)

# print(cur_path)
# # new_path = os.path.join('pages', 'resources', 'df_imputed_120422.csv')
#
# new_path = os.path.relpath('../../process/df_imputed_120422.csv', cur_path)
# # print(new_path)
#
# df = pd.read_csv(new_path, infer_datetime_format=True)
#
# df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], format="%Y-%m-%d %H:%M:%S")
#
# df.index = df['TimeStamp']
#
# df.index.sort_values()
#
# aq_df = df
#
# create_models('pm25')
# create_models('pm1')
# create_models('pm10')

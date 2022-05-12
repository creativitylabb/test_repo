import pandas as pd

from one_pagination import scroll_page

from datetime import datetime

# df = pd.DataFrame()
#
# pagination_df = scroll_page('brasov-dev', '1m', df)

# pagination_df.to_csv('air_dataset.csv')


df = pd.read_csv('air_dataset.csv')

df.drop('Unnamed: 0', axis=1, inplace=True)

for sensor_name in df['Sensor'].unique():
    df[sensor_name] = df['Value'][df['Sensor'] == sensor_name]

df = df.drop(['index', 'type', 'Sensor', 'Value', 'id', 'score'], axis=1)

# df['TimeStamp'] = 1000*pd.to_datetime(df['TimeStamp'], format="%Y-%m-%d %H:%M:%S")
df['TimeStamp'] = df['TimeStamp'].apply(lambda x: datetime.utcfromtimestamp(x / 1000).strftime('%Y-%m-%d %H:%M:%S'))
# df = df.drop(['index', 'type','Sensor','Value','id','score'], axis=1)


gk = df.groupby('TimeStamp')

# Let's print the first entries
# in all the groups formed.
print(gk.first())
gk.first().to_csv('last_step_pagination_110422.csv')


#
# import pandas as pd
# from datetime import datetime
# #
# df=pd.read_csv('air_quality_final.csv')
# df.drop('Unnamed: 0', axis=1, inplace=True)
# print(len(df))
#
# df['TimeStamp']=df['TimeStamp'].apply(lambda x: datetime.utcfromtimestamp(x / 1000).strftime('%Y-%m-%d %H:%M:%S'))
#
# print(len(df))
# df.to_csv('air_quality_final_final.csv')

# df['TimeStamp']=df['TimeStamp'].apply(lambda x: datetime.utcfromtimestamp(x / 1000).strftime('%Y-%m-%d %H:%M:%S'))
# df=df.sort_values(by="TimeStamp", key=pd.to_datetime)

# df.sort_values(by='TimeStamp')
# print(df['TimeStamp'])
# df.to_csv('timestamp.csv')
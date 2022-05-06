import pandas as pd
from datetime import datetime

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
gk.first().to_csv('air_quality_final.csv')

# print(df.head())


# df.to_csv('air_quality_final.csv')

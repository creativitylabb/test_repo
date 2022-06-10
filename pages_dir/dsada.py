import pandas as pd
import os
clean_df = pd.read_csv('last_step_pagination_110422.csv', infer_datetime_format=True)

# df = df.drop('Unnamed: 0', axis=1)

clean_df['TimeStamp'] = pd.to_datetime(clean_df['TimeStamp'])
print(clean_df)
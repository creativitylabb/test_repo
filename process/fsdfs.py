import pandas as pd
df=pd.read_csv('df_imputed_120422.csv',infer_datetime_format=True)
df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], format="%Y-%m-%d %H:%M:%S")

clean_df = df.resample('24h', on='TimeStamp').mean()
print(clean_df)

from darts.datasets import AirPassengersDataset, MonthlyMilkDataset
import matplotlib.pyplot as plt
from darts.dataprocessing.transformers import Scaler
from darts.models import NBEATSModel
from darts.metrics import mape, smape

# df=AirPassengersDataset().load().pd_dataframe()
# print(df)

series_air = AirPassengersDataset().load()
series_air_df = AirPassengersDataset().load().pd_dataframe()

print(series_air_df)

series_milk = MonthlyMilkDataset().load()
series_milk_df = MonthlyMilkDataset().load().pd_dataframe()
print(series_air_df.columns)

series_air.plot(label='Number of air passengers')
series_milk.plot(label='Pounds of milk produced per cow')
plt.legend()
plt.show()

# standard scaling->plot in the same scale
scaler_air, scaler_milk = Scaler(), Scaler()
series_air_scaled = scaler_air.fit_transform(series_air)
series_milk_scaled = scaler_milk.fit_transform(series_milk)

# plot in the same scale
series_air_scaled.plot(label='air')
series_milk_scaled.plot(label='milk')
plt.legend()
plt.show()

# Train And Validation Split
# first 36->train
# first 36->test
train_air, val_air = series_air_scaled[:-36], series_air_scaled[-36:]
train_milk, val_milk = series_milk_scaled[:-36], series_milk_scaled[-36:]

#deep learning model
#input_chunk_length = take first 24 data as train
#next 12 will be output
#move like window->next 24 data will be input and so on... (sliding window)?
model_air_milk = NBEATSModel(input_chunk_length=24, output_chunk_length=12, n_epochs=100, random_state=0)

model_air_milk.fit([train_air, train_milk], verbose=True)

# predict for next 36 months
pred = model_air_milk.predict(n=36, series=train_air)
# pred = model_air_milk.predict(n=36, series=train_air,past_covariates=...) #to add other data for multivariate analysis

series_air_scaled.plot(label='actual')
pred.plot(label='forecast')
plt.legend()
plt.show()

print('MAPE = {:.2f}%'.format(mape(series_air_scaled, pred)))

pred = model_air_milk.predict(n=36, series=train_milk)

series_milk_scaled.plot(label='actual')
pred.plot(label='forecast')
plt.legend()
plt.show()

print('MAPE = {:.2f}%'.format(mape(series_milk_scaled, pred)))

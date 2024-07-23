import pandas as pd
import darts
from darts.models import ExponentialSmoothing

datafile = pd.read_csv("AirPassengers.csv", delimiter = ",")
print(datafile)
series = darts.TimeSeries.from_dataframe(datafile, "Month", "#Passengers")
train, val = series[:-36], series[-36:]
model = ExponentialSmoothing()
model.fit(train)
prediction = model.predict(len(val), num_samples = 1000)
print(prediction)
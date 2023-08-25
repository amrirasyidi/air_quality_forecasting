# Data Source
1. [MCC-SINICA](https://github.com/MCC-SINICA/Using-Satellite-Data-on-Remote-Transportation)
2. [aqicn.org](https://aqicn.org/historical#!city:indonesia/jakarta/us-consulate/south)
3. [openaq](https://openaq.org/developers/platform-overview/)

# Notes

## forecast testing

Using params as follows:
```
history_len = 48
batch_size = 8 
epochs = 10
lr = 1e-3
weight_decay = 0.01
```

The testing model perform quite well on the first x hours of the forecasting as can be seen in the graph below:
![Forecasting Result](notebooks\images\prediction_result_24_hour.png "Prediction of first 24 hour")

![Forecasting Result](notebooks\images\prediction_result_100_hour.png "Prediction of first 100 hour")

## epa_tw exploration

tbd

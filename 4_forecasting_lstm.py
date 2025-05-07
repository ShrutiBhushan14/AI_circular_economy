
# Prepare time series data for forecasting
zone_data = df.groupby(['date', 'zone'])['weight'].sum().reset_index()
zone_data['date'] = pd.to_datetime(zone_data['date'])
zone_data = zone_data.pivot(index='date', columns='zone', values='weight').fillna(0)

# For simplicity, use zone 0
series = zone_data[0].values
n_input = 10
n_features = 1
generator = TimeseriesGenerator(series, series, length=n_input, batch_size=1)

model_lstm = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, activation='relu', input_shape=(n_input, n_features)),
    tf.keras.layers.Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(generator, epochs=20)

# Forecast next 7 days
pred_input = series[-n_input:]
forecast = []
for _ in range(7):
    input_reshaped = pred_input.reshape((1, n_input, 1))
    yhat = model_lstm.predict(input_reshaped, verbose=0)
    forecast.append(yhat[0][0])
    pred_input = np.append(pred_input[1:], yhat)

plt.plot(range(len(series)), series, label='History')
plt.plot(range(len(series), len(series)+7), forecast, label='Forecast')
plt.legend()
plt.title('7-Day Waste Forecast (Zone 0)')
plt.show()

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# 1. Download Bitcoin historical data
data = yf.download(
    "BTC-USD",
    start="2018-01-01",
    end="2026-01-01"
)

# 2. Select closing price
prices = data[["Close"]].dropna()

# 3. Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)


# 4. Create sequences
def create_dataset(dataset, time_step=60):
    X = []
    y = []

    for i in range(time_step, len(dataset)):
        X.append(dataset[i-time_step:i, 0])
        y.append(dataset[i, 0])

    return np.array(X), np.array(y)


time_step = 60

X, y = create_dataset(scaled_prices, time_step)


# 5. Reshape data for LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)


# 6. Split into training and testing data
train_size = int(len(X) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]


# 7. Build LSTM model
model = Sequential()

model.add(
    LSTM(
        50,
        return_sequences=True,
        input_shape=(time_step, 1)
    )
)

model.add(Dropout(0.2))

model.add(
    LSTM(
        50,
        return_sequences=False
    )
)

model.add(Dropout(0.2))

model.add(Dense(25))
model.add(Dense(1))


# 8. Compile model
model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)


# 9. Train model
print("Training model...")

model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    verbose=1
)


# 10. Make predictions
print("Making predictions...")

predictions = model.predict(X_test)


# 11. Convert values back to USD
predictions = scaler.inverse_transform(predictions)

actual_prices = scaler.inverse_transform(
    y_test.reshape(-1, 1)
)


# 12. Calculate RMSE and MAE
rmse = np.sqrt(
    mean_squared_error(
        actual_prices,
        predictions
    )
)

mae = mean_absolute_error(
    actual_prices,
    predictions
)


print("\n========== RESULTS ==========")
print("RMSE:", rmse)
print("MAE :", mae)


# 13. Plot actual vs predicted prices
plt.figure(figsize=(12, 6))

plt.plot(
    actual_prices,
    label="Actual Price"
)

plt.plot(
    predictions,
    label="Predicted Price"
)

plt.title(
    "Bitcoin Price Prediction Using LSTM"
)

plt.xlabel("Time")
plt.ylabel("Price in USD")

plt.legend()
plt.show()
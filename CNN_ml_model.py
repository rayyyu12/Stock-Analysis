import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import os
import re
import sys

def find_preprocessed_file():
    """
    Find the latest preprocessed CSV file in the preprocessed_data directory.

    Returns:
        tuple: A tuple containing the file path and ticker symbol if found; otherwise, (None, None).
    """
    preprocessed_data_folder = os.path.join(os.getcwd(), 'preprocessed_data')  # Path to the preprocessed_data folder
    csv_files = [f for f in os.listdir(preprocessed_data_folder) if f.endswith('.csv')]

    if not csv_files:
        print("Error: No CSV files found in the preprocessed_data directory.")
        return None, None

    # Assume the latest file is the one with the most recent modification time
    csv_files_full_path = [os.path.join(preprocessed_data_folder, f) for f in csv_files]
    latest_file = max(csv_files_full_path, key=os.path.getmtime)

    # Extract ticker symbol from the filename using regular expressions
    filename = os.path.basename(latest_file)
    match = re.match(r"([A-Za-z]+)_processed\.csv", filename)
    if match:
        ticker = match.group(1)
        return latest_file, ticker
    else:
        print("Error: Could not extract ticker symbol from the filename.")
        return None, None

# Find the preprocessed data file and extract the ticker symbol
file_path, stock_ticker = find_preprocessed_file()

if file_path is None or stock_ticker is None:
    print("Error: No valid preprocessed CSV file found to process.")
    sys.exit(1)
else:
    print(f"Loading data for ticker '{stock_ticker}' from file '{file_path}'")

# Load your preprocessed data
data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
data = data[['close', 'SMA']]  # Use relevant columns

# Ensure there are no NaN values
data.dropna(inplace=True)

# Verify that dates are parsed correctly
print(f"Data index type: {type(data.index)}")
print(f"Last date in data: {data.index[-1]}")

# Initialize scalers for 'close' and 'SMA'
scaler_close = MinMaxScaler()
scaler_sma = MinMaxScaler()

# Fit scalers and transform the data
data['close'] = scaler_close.fit_transform(data[['close']])
data['SMA'] = scaler_sma.fit_transform(data[['SMA']])

# Convert data to NumPy array
scaled_data = data.values

# Function to create sequences
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), :])  # Use all features
        Y.append(data[i + time_step, 0])      # Predict 'close' price
    return np.array(X), np.array(Y)

time_step = 10  # Number of previous days to consider
X, Y = create_dataset(scaled_data, time_step)

# Split into training and test sets (use shuffle=False to maintain time order)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, shuffle=False)

# Define the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu',
                           input_shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(filters=128, kernel_size=2, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Use EarlyStopping to prevent overfitting
from tensorflow.python.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, Y_train,
    epochs=1000,
    batch_size=10,
    validation_data=(X_test, Y_test),
    callbacks=[early_stopping]
)

# Evaluate the model
train_loss = model.evaluate(X_train, Y_train)
test_loss = model.evaluate(X_test, Y_test)

print(f"Training Loss: {train_loss}")
print(f"Test Loss: {test_loss}")

# Make predictions
predicted_prices = model.predict(X_test)
predicted_prices = scaler_close.inverse_transform(predicted_prices)

# Inverse transform the actual prices
Y_test_actual = scaler_close.inverse_transform(Y_test.reshape(-1, 1))

# Flatten the arrays
y_pred = predicted_prices.flatten()
y_true = Y_test_actual.flatten()

# Calculate metrics
mae = mean_absolute_error(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
r2 = r2_score(y_true, y_pred)

# Print metrics
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"R-squared Score: {r2:.4f}")

# Visualize the results
plt.figure(figsize=(14, 7))
plt.plot(data.index[-len(y_true):], y_true, label='Actual Close Price')
plt.plot(data.index[-len(y_pred):], y_pred, label='Predicted Close Price')
plt.title(f'Actual vs Predicted Close Price for {stock_ticker}')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.tight_layout()
# Save the plot
output_folder = os.path.join(os.getcwd(), 'outputs')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
ml_graph_path = os.path.join(output_folder, f"{stock_ticker}_ml_graph.png")
plt.savefig(ml_graph_path)
plt.close()
print(f"ML graph saved to {ml_graph_path}")

# --- Additions start here ---

# Forecast stock prices for the next week (7 days)
forecast_horizon = 7  # Number of days to forecast

# Get the last 'time_step' days from the data to start forecasting
last_sequence = scaled_data[-time_step:]
forecast_input = last_sequence.copy()

# List to store the forecasted prices
forecasted_prices = []

for _ in range(forecast_horizon):
    # Prepare the input data
    input_data = forecast_input.reshape(1, time_step, -1)
    # Predict the next day's 'close' price
    pred_price_scaled = model.predict(input_data)
    # Append the predicted price to the forecasted prices list
    forecasted_prices.append(pred_price_scaled[0, 0])
    # Create the next input sequence
    next_input = np.append(forecast_input[1:], [[pred_price_scaled[0, 0], forecast_input[-1, 1]]], axis=0)
    forecast_input = next_input

# Inverse transform the forecasted prices
forecasted_prices = np.array(forecasted_prices).reshape(-1, 1)
forecasted_prices_inversed = scaler_close.inverse_transform(forecasted_prices).flatten()

# Create date range for future predictions
future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon, freq='B')

future_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted Close Price': forecasted_prices_inversed
})
future_df.set_index('Date', inplace=True)

# Print future predictions
print("\nForecasted Close Prices for the Next Week:")
print(future_df)

# Save future predictions to CSV
forecast_csv_path = os.path.join(output_folder, f"{stock_ticker}_future_predictions.csv")
future_df.to_csv(forecast_csv_path)
print(f"Future predictions saved to {forecast_csv_path}")

# Plot the forecasted results
plt.figure(figsize=(14, 7))

# Plot historical data (actual close prices)
historical_prices = scaler_close.inverse_transform(data['close'].values.reshape(-1, 1)).flatten()
plt.plot(data.index, historical_prices, label='Historical Close Price')

# Plot future predictions
plt.plot(future_dates, forecasted_prices_inversed, label='Forecasted Close Price', marker='o', linestyle='--')

plt.title(f'Stock Price Forecast for {stock_ticker} for the Next Week')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
forecast_graph_path = os.path.join(output_folder, f"{stock_ticker}_forecast_graph.png")
plt.savefig(forecast_graph_path)
plt.close()
print(f"Forecast graph saved to {forecast_graph_path}")

# --- Additions end here ---

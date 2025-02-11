import pandas as pd
import yfinance as yf
import os  # Import os to handle directory paths
from datetime import datetime

def fetch_historical_data(symbol):
    # Set the start date to January 1, 2024
    start_date = "2024-01-01"
    # Set the end date to today
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    # Fetch historical data for the specified symbol
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    
    if not data.empty:
        return data
    else:
        print("No valid historical data retrieved.")
        return None

def preprocess_data(data):
    print("Columns in DataFrame:", data.columns)

    if data.shape[1] >= 5:
        # Adjust column names based on the data retrieved
        data = data.reset_index()
        data.columns = [col.lower().replace(' ', '_') for col in data.columns]
    else:
        print("Unexpected number of columns:", data.shape[1])
        return None

    data = data.astype(float, errors='ignore')
    data = data[['date', 'open', 'high', 'low', 'close', 'volume']]
    
    return data

def save_to_csv(data, filename):
    # Create csv_data folder if it doesn't exist
    folder_name = 'csv_data'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)  # Create the folder

    # Save the DataFrame to a CSV file in the csv_data folder
    file_path = os.path.join(folder_name, filename)
    data.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

def main():
    # Prompt the user for the stock ticker symbol
    stock_symbol = input("Please enter the stock ticker symbol (e.g., AAPL): ").strip().upper()
    historical_data = fetch_historical_data(stock_symbol)

    if historical_data is not None:
        print("Data retrieved successfully.")
        processed_data = preprocess_data(historical_data)
        if processed_data is not None:
            save_to_csv(processed_data, f"{stock_symbol}_historical_data.csv")  # Save to CSV
        else:
            print("Data preprocessing failed.")
    else:
        print("No valid historical data retrieved.")

if __name__ == "__main__":
    main()

import pandas as pd
import os  # Import os to navigate the file system
import sys  # Import sys to handle command-line arguments
import re   # Import re for regular expression matching

def load_data(file_path):
    """
    Load stock data from a CSV file.

    Parameters:
        file_path (str): The path to the CSV file containing stock data.

    Returns:
        pd.DataFrame: DataFrame containing stock data.
    """
    try:
        data = pd.read_csv(file_path, index_col='date', parse_dates=True)
        print("Data loaded successfully.")
        print("Columns in the dataset:", data.columns)  # Print the column names
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    return data

def calculate_moving_average(data, window=5):
    """
    Calculate simple moving average.

    Parameters:
        data (pd.DataFrame): DataFrame containing stock data.
        window (int): The window size for moving average.

    Returns:
        pd.Series: Series containing the moving average.
    """
    closing_column_name = 'close'  # Ensure the correct column name
    if closing_column_name not in data.columns:
        raise KeyError(f"The '{closing_column_name}' column is not present in the data.")
    return data[closing_column_name].rolling(window=window).mean()

def preprocess_data(file_path):
    """
    Main function to load and preprocess stock data.

    Parameters:
        file_path (str): The path to the CSV file containing stock data.

    Returns:
        pd.DataFrame: Processed DataFrame ready for ML model.
    """
    # Load data
    data = load_data(file_path)
    if data is None:
        return None  # Exit if data loading failed

    # Add moving average
    data['SMA'] = calculate_moving_average(data)

    # Drop rows with NaN values (due to SMA calculation)
    data.dropna(inplace=True)

    return data

def find_latest_csv_file():
    """
    Find the latest CSV file in the csv_data directory.

    Returns:
        tuple: A tuple containing the file path and ticker symbol if found; otherwise, (None, None).
    """
    csv_data_folder = os.path.join(os.getcwd(), 'csv_data')  # Path to the csv_data folder
    csv_files = [f for f in os.listdir(csv_data_folder) if f.endswith('.csv')]

    if not csv_files:
        print("Error: No CSV files found in the csv_data directory.")
        return None, None

    # Assume the latest file is the one with the most recent modification time
    csv_files_full_path = [os.path.join(csv_data_folder, f) for f in csv_files]
    latest_file = max(csv_files_full_path, key=os.path.getmtime)

    # Extract ticker symbol from the filename using regular expressions
    filename = os.path.basename(latest_file)
    match = re.match(r"([A-Za-z]+)_historical_data\.csv", filename)
    if match:
        ticker = match.group(1)
        return latest_file, ticker
    else:
        print("Error: Could not extract ticker symbol from the filename.")
        return None, None

def save_processed_data(data, ticker):
    """
    Save processed data to the preprocessed_data folder.

    Parameters:
        data (pd.DataFrame): DataFrame containing processed stock data.
        ticker (str): The ticker symbol for naming the file.
    """
    preprocessed_data_folder = os.path.join(os.getcwd(), 'preprocessed_data')  # Path to the preprocessed_data folder
    if not os.path.exists(preprocessed_data_folder):
        os.makedirs(preprocessed_data_folder)  # Create the folder if it doesn't exist

    output_file_path = os.path.join(preprocessed_data_folder, f"{ticker}_processed.csv")  # Construct output file path
    data.to_csv(output_file_path)
    print(f"Processed data saved to {output_file_path}")

if __name__ == "__main__":
    # Automatically find the latest CSV file and extract the ticker symbol
    file_path, ticker = find_latest_csv_file()

    if file_path is None or ticker is None:
        print("Error: No valid CSV file found to process.")
        sys.exit(1)
    else:
        print(f"Processing data for ticker '{ticker}' from file '{file_path}'")
        # Preprocess the data
        processed_data = preprocess_data(file_path)
        if processed_data is not None:
            # Output processed data to a new CSV file
            save_processed_data(processed_data, ticker)
        else:
            print("Data preprocessing failed.")
            sys.exit(1)

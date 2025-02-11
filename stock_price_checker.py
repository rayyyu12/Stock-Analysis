import requests
import json

def fetch_stock_data(symbol, api_key):
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching stock data: {e}")
        return None

def main():
    api_key = "M4O9KCL4GNM8IVJW"  # Replace with your API key
    stock_symbol = "VOO"  # Example stock symbol (you can change this)
    stock_data = fetch_stock_data(stock_symbol, api_key)

    if stock_data:
        try:
            # Parse the JSON data
            json_data = json.loads(stock_data)
            
            # Access the stock price from the JSON structure
            price = json_data["Global Quote"]["05. price"]
            print(f"Stock Price for {stock_symbol}: {price}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing stock data: {e}")
    else:
        print(f"No data received for stock symbol: {stock_symbol}")

if __name__ == "__main__":
    main()

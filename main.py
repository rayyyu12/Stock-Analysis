import os
import sys
import subprocess

def stock_price_checker():
    # Run stock_price_checker.py
    try:
        subprocess.run(['python', 'stock_price_checker.py'], check=True)
    except subprocess.CalledProcessError:
        print("Failed to retrieve stock price.")
        sys.exit(1)

def price_prediction_and_report():
    # Run data_collection.py
    try:
        subprocess.run(['python', 'data_collection.py'], check=True)
    except subprocess.CalledProcessError:
        print("Data collection failed.")
        sys.exit(1)

    # Run data_preprocessing.py
    try:
        subprocess.run(['python', 'data_preprocessing.py'], check=True)
    except subprocess.CalledProcessError:
        print("Data preprocessing failed.")
        sys.exit(1)

    # Run CNN_ml_model.py
    try:
        subprocess.run(['python', 'CNN_ml_model.py'], check=True)
    except subprocess.CalledProcessError:
        print("Machine learning model failed.")
        sys.exit(1)

    # Run stock_news_analysis.py
    try:
        subprocess.run(['python', 'news_sentiment.py'], check=True)
    except subprocess.CalledProcessError:
        print("News sentiment analysis failed.")
        sys.exit(1)

    # Run generate_report.py
    try:
        subprocess.run(['python', 'generate_report.py'], check=True)
    except subprocess.CalledProcessError:
        print("Report generation failed.")
        sys.exit(1)

    print("Price prediction report generated. Please check the 'reports' directory.")

def main():
    print("Welcome to the Stock Analysis Tool!")
    print("Please select an option:")
    print("1. Stock Price Checker")
    print("2. Price Prediction and Report Generation")
    choice = input("Enter your choice (1 or 2): ")

    if choice not in ('1', '2'):
        print("Invalid choice. Exiting.")
        sys.exit(1)

    if choice == '1':
        stock_price_checker()
    elif choice == '2':
        price_prediction_and_report()

if __name__ == '__main__':
    main()

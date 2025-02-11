# Stock Analysis Tool 📈

A comprehensive Python-based tool for stock analysis, price prediction, and sentiment analysis. This tool combines machine learning (CNN) with technical analysis and news sentiment to provide detailed stock insights and future price predictions.

## Features 🚀

- **Real-time Stock Price Checking**: Fetch current stock prices using Yahoo Finance API
- **Historical Data Analysis**: Collect and analyze historical stock data
- **Price Prediction**: Implement CNN (Convolutional Neural Network) for stock price forecasting
- **Technical Analysis**: Calculate technical indicators like Simple Moving Average (SMA)
- **News Sentiment Analysis**: Analyze news sentiment for more informed decision making
- **Automated Report Generation**: Create comprehensive PDF reports with graphs and predictions

## Prerequisites 📋

Before running this tool, make sure you have Python 3.x installed and the following dependencies:

```bash
pip install pandas
pip install numpy
pip install tensorflow
pip install yfinance
pip install scikit-learn
pip install matplotlib
pip install reportlab
```

## Project Structure 📁

```
Stock-Analysis/
├── csv_data/                  # Storage for historical data CSV files
├── preprocessed_data/         # Processed data ready for ML model
├── outputs/                   # Generated graphs and predictions
├── reports/                   # PDF reports
├── data_collection.py         # Historical data fetching script
├── data_preprocessing.py      # Data preprocessing script
├── CNN_ml_model.py           # Machine learning model implementation
├── news_sentiment.py         # News sentiment analysis script
├── generate_report.py        # PDF report generation script
└── main.py                   # Main execution script
```

## Usage 🔧

1. Clone the repository:
```bash
git clone https://github.com/rayyyu12/Stock-Analysis.git
cd Stock-Analysis
```

2. Run the main script:
```bash
python main.py
```

3. Choose your desired option:
   - Option 1: Quick stock price check
   - Option 2: Generate comprehensive analysis report

## Features in Detail 🔍

### Stock Price Checker
- Real-time stock price retrieval
- Basic stock information display

### Price Prediction and Report Generation
1. **Data Collection**
   - Fetches historical data from Yahoo Finance
   - Stores raw data in CSV format

2. **Data Preprocessing**
   - Cleans and normalizes data
   - Calculates technical indicators
   - Prepares data for ML model

3. **Machine Learning Model**
   - Implements CNN for price prediction
   - Uses historical prices and technical indicators
   - Provides 7-day price forecasts

4. **News Sentiment Analysis**
   - Analyzes recent news sentiment
   - Provides sentiment scores and summaries

5. **Report Generation**
   - Creates comprehensive PDF reports
   - Includes graphs, predictions, and sentiment analysis
   - Automated formatting and organization

## Output 📊

The tool generates several outputs:
- ML model performance graphs
- Price prediction forecasts
- Sentiment analysis results
- Comprehensive PDF reports

## Contributing 🤝

Contributions, issues, and feature requests are welcome! Feel free to check [issues page](https://github.com/rayyyu12/Stock-Analysis/issues).

## Acknowledgments 💡

- Yahoo Finance for providing financial data
- TensorFlow team for the ML framework
- All contributors and maintainers
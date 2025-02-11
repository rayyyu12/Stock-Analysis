# stock_news_analysis.py

import os
import sys
import time
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from newsapi import NewsApiClient
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from dateutil.relativedelta import relativedelta
import yfinance as yf

# Ensure NLTK resources are downloaded
nltk.download('vader_lexicon', quiet=True)

def get_news_articles(api_key, query, from_date, to_date, language='en', max_articles=20):
    """
    Fetch news articles related to the query using NewsAPI within a specified date range.
    """
    newsapi = NewsApiClient(api_key=api_key)
    all_articles = []
    page = 1
    page_size = 100  # Maximum allowed by NewsAPI

    while len(all_articles) < max_articles:
        try:
            response = newsapi.get_everything(
                q=query,  # Search in the entire content
                from_param=from_date,
                to=to_date,
                language=language,
                sort_by='relevancy',
                page_size=page_size,
                page=page
            )
            articles = response.get('articles', [])
            if not articles:
                break
            all_articles.extend(articles)
            if len(articles) < page_size:
                break  # No more articles to fetch
            page += 1
            time.sleep(1)  # Respect rate limits
        except Exception as e:
            print(f"An error occurred while fetching articles: {e}")
            break

    # Limit to the max_articles specified
    return all_articles[:max_articles]

def extract_text_from_articles(articles):
    """
    Extract the description or content from news articles.
    """
    texts = []
    for article in articles:
        # Prefer description over content if available
        text = article.get('description') or article.get('content')
        if text:
            texts.append(text)
    return texts

def preprocess_text(text):
    """
    Preprocess the text for analysis.

    Parameters:
        text (str): The text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    # Remove URLs, mentions, hashtags, and special characters
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

def analyze_sentiment(texts, batch_size=8):
    """
    Analyze the sentiment of a list of texts using a pre-trained transformer model.

    Parameters:
        texts (list): List of texts to analyze.
        batch_size (int): Number of texts to process in a batch.

    Returns:
        list: List of sentiment scores.
    """
    if not texts:
        return []

    # Use GPU if available, otherwise fallback to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load a lighter pre-trained model and tokenizer
    model_name = 'distilbert-base-uncased-finetuned-sst-2-english'  # Binary sentiment model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)

    sentiment_scores = []

    # Process texts in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        scores = probabilities.cpu().tolist()
        sentiment_scores.extend(scores)

    return sentiment_scores

def aggregate_sentiment(sentiment_scores, articles):
    """
    Aggregate sentiment scores to provide overall insight.
    """
    if not sentiment_scores or not articles:
        return {
            'positive_percentage': 0,
            'negative_percentage': 0,
            'most_positive_article': '',
            'most_negative_article': '',
            'most_positive_url': '',
            'most_negative_url': '',
        }

    positive_count = 0
    negative_count = 0
    positive_articles = []
    negative_articles = []

    for score, article in zip(sentiment_scores, articles):
        positive_score = score[1]
        negative_score = score[0]
        text = article.get('description') or article.get('content', '')
        url = article.get('url', '')
        if positive_score > negative_score:
            positive_count += 1
            positive_articles.append((positive_score, text, url))
        else:
            negative_count += 1
            negative_articles.append((negative_score, text, url))

    total = positive_count + negative_count
    positive_percentage = (positive_count / total) * 100 if total > 0 else 0
    negative_percentage = (negative_count / total) * 100 if total > 0 else 0

    # Get the most positive and negative articles
    most_positive = max(positive_articles, default=(0, "", ""), key=lambda x: x[0])
    most_negative = max(negative_articles, default=(0, "", ""), key=lambda x: x[0])

    return {
        'positive_percentage': positive_percentage,
        'negative_percentage': negative_percentage,
        'most_positive_article': most_positive[1],
        'most_negative_article': most_negative[1],
        'most_positive_url': most_positive[2],
        'most_negative_url': most_negative[2],
    }

def provide_insight(aggregate_results):
    """
    Provide insight based on aggregated sentiment results.

    Parameters:
        aggregate_results (dict): Aggregated sentiment results.

    Returns:
        str: Insight or prediction about the stock price.
    """
    positive_percentage = aggregate_results['positive_percentage']
    negative_percentage = aggregate_results['negative_percentage']

    insight = f"Out of the analyzed articles, {positive_percentage:.1f}% are positive and {negative_percentage:.1f}% are negative.\n"

    if positive_percentage > negative_percentage:
        insight += "Overall positive sentiment detected. The stock price may increase."
    elif negative_percentage > positive_percentage:
        insight += "Overall negative sentiment detected. The stock price may decrease."
    else:
        insight += "Neutral sentiment detected. The stock price may remain stable."

    return insight

def find_ticker_from_data():
    """
    Find the latest preprocessed CSV file and extract the ticker symbol.
    """
    preprocessed_data_folder = os.path.join(os.getcwd(), 'preprocessed_data')
    csv_files = [f for f in os.listdir(preprocessed_data_folder) if f.endswith('.csv')]

    if not csv_files:
        print("Error: No CSV files found in the preprocessed_data directory.")
        return None

    # Assume the latest file is the one with the most recent modification time
    csv_files_full_path = [os.path.join(preprocessed_data_folder, f) for f in csv_files]
    latest_file = max(csv_files_full_path, key=os.path.getmtime)

    # Extract ticker symbol from the filename using regular expressions
    filename = os.path.basename(latest_file)
    match = re.match(r"([A-Za-z]+)_processed\.csv", filename)
    if match:
        ticker = match.group(1)
        return ticker
    else:
        print("Error: Could not extract ticker symbol from the filename.")
        return None

def get_company_name(ticker):
    """
    Retrieve the company name using yfinance.
    """
    stock = yf.Ticker(ticker)
    return stock.info.get('shortName', '')

def main():
    # Automatically find the ticker symbol from the preprocessed data
    ticker = find_ticker_from_data()
    if not ticker:
        print("Error: Could not determine the stock ticker symbol.")
        sys.exit(1)
    else:
        print(f"Using ticker symbol: {ticker}")

    # Get the company name
    company_name = get_company_name(ticker)
    if not company_name:
        company_name = ticker  # Fallback to ticker if company name not found
        print(f"Warning: Could not retrieve company name for {ticker}. Using ticker symbol in search.")
    else:
        print(f"Company name for {ticker} is {company_name}")

    # Adjust the query
    query = f'("{company_name}" OR "{ticker}") AND (stock OR stocks OR "stock price" OR earnings)'

    # Get your NewsAPI key from environment variable
    api_key = '1cef28f52b914df4b35f99e70ad9401b'
    if not api_key:
        print("Error: NEWSAPI_KEY environment variable not set.")
        sys.exit(1)

    # Set default date range (last 1 month)
    from_date = (datetime.now() - relativedelta(months=1)).strftime('%Y-%m-%d')
    to_date = datetime.now().strftime('%Y-%m-%d')
    print(f"Date range: {from_date} to {to_date}")

    # Limit the number of articles to process
    max_articles = 20

    # Fetch news articles
    print(f"Fetching news articles for {company_name} ({ticker}) from {from_date} to {to_date}...")
    articles = get_news_articles(api_key, query, from_date=from_date, to_date=to_date, max_articles=max_articles)

    if not articles:
        print("No articles found. Proceeding with empty sentiment analysis.")
        sentiment_scores = []
        aggregate_results = {
            'positive_percentage': 0,
            'negative_percentage': 0,
            'most_positive_article': '',
            'most_negative_article': '',
            'most_positive_url': '',
            'most_negative_url': '',
        }
    else:
        print(f"Processing {len(articles)} articles.")

        # Extract text from articles
        print("Extracting text from articles...")
        texts = extract_text_from_articles(articles)

        # Preprocess texts
        print("Preprocessing texts...")
        preprocessed_texts = [preprocess_text(text) for text in texts]

        # Analyze sentiment
        print("Analyzing sentiment...")
        sentiment_scores = analyze_sentiment(preprocessed_texts)

        # Aggregate sentiment results
        print("Aggregating sentiment scores...")
        aggregate_results = aggregate_sentiment(sentiment_scores, articles)

    # Provide insight
    insight = provide_insight(aggregate_results)

    # Save sentiment analysis results to a text file
    output_folder = os.path.join(os.getcwd(), 'outputs')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    sentiment_file_path = os.path.join(output_folder, f"{ticker}_sentiment.txt")
    with open(sentiment_file_path, 'w', encoding='utf-8') as f:
        f.write("--- Sentiment Analysis Result ---\n")
        f.write(f"Positive Articles: {aggregate_results['positive_percentage']:.1f}%\n")
        f.write(f"Negative Articles: {aggregate_results['negative_percentage']:.1f}%\n")
        f.write(f"Insight: {insight}\n\n")
        if aggregate_results['most_positive_article']:
            f.write("Most Positive Article Excerpt:\n")
            f.write(aggregate_results['most_positive_article'] + "\n\n")
            f.write("Most Positive Article URL:\n")
            f.write(aggregate_results['most_positive_url'] + "\n\n")
        if aggregate_results['most_negative_article']:
            f.write("Most Negative Article Excerpt:\n")
            f.write(aggregate_results['most_negative_article'] + "\n\n")
            f.write("Most Negative Article URL:\n")
            f.write(aggregate_results['most_negative_url'] + "\n\n")

    print(f"\nSentiment analysis results saved to {sentiment_file_path}")

if __name__ == "__main__":
    main()

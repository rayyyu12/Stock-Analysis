import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
import re

def find_ticker():
    """
    Find the ticker symbol from the output files.
    """
    outputs_folder = os.path.join(os.getcwd(), 'outputs')
    files = os.listdir(outputs_folder)
    for file in files:
        match = re.match(r"([A-Za-z]+)_ml_graph\.png", file)
        if match:
            ticker = match.group(1)
            return ticker
    print("Error: Could not find the ticker symbol from output files.")
    return None

def create_pdf(stock_ticker):
    # Ensure 'reports' directory exists
    reports_folder = os.path.join(os.getcwd(), 'reports')
    if not os.path.exists(reports_folder):
        os.makedirs(reports_folder)

    pdf_file = os.path.join(reports_folder, f"{stock_ticker}_report.pdf")
    ml_graph = os.path.join(os.getcwd(), 'outputs', f"{stock_ticker}_ml_graph.png")
    forecast_graph = os.path.join(os.getcwd(), 'outputs', f"{stock_ticker}_forecast_graph.png")
    sentiment_file = os.path.join(os.getcwd(), 'outputs', f"{stock_ticker}_sentiment.txt")
    forecast_csv = os.path.join(os.getcwd(), 'outputs', f"{stock_ticker}_future_predictions.csv")

    # Read sentiment analysis text
    with open(sentiment_file, 'r', encoding='utf-8') as f:
        sentiment_text = f.read()

    doc = SimpleDocTemplate(pdf_file, pagesize=letter)
    styles = getSampleStyleSheet()
    flowables = []

    # Add title
    flowables.append(Paragraph(f"Price Prediction Report for {stock_ticker}", styles['Title']))
    flowables.append(Spacer(1, 12))

    # Add ML graph
    flowables.append(Paragraph("Machine Learning Model Results:", styles['Heading2']))
    flowables.append(Spacer(1, 12))
    flowables.append(Image(ml_graph, width=400, height=200))
    flowables.append(Spacer(1, 12))

    # Add Forecast graph
    flowables.append(Paragraph("Stock Price Forecast for the Next Week:", styles['Heading2']))
    flowables.append(Spacer(1, 12))
    flowables.append(Image(forecast_graph, width=400, height=200))
    flowables.append(Spacer(1, 24))  # Add a bit more space under the forecast graph
    flowables.append(PageBreak())  # Ensure the table starts on a new page

    # Add future price predictions table
    flowables.append(Paragraph("Future Price Predictions:", styles['Heading2']))
    flowables.append(Spacer(1, 12))
    # Call function to add the table
    flowables.extend(create_forecast_table(forecast_csv))

    # Add sentiment analysis
    flowables.append(Paragraph("Sentiment Analysis:", styles['Heading2']))
    flowables.append(Spacer(1, 12))
    flowables.append(Paragraph(sentiment_text.replace('\n', '<br />'), styles['Normal']))

    doc.build(flowables)
    print(f"Report generated: {pdf_file}")

def create_forecast_table(csv_file):
    from reportlab.platypus import Table, TableStyle
    from reportlab.lib import colors

    # Read the CSV file into a list of lists
    import csv
    table_data = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        table_data.append(headers)
        for row in reader:
            table_data.append(row)

    # Create a Table object
    table = Table(table_data)

    # Add style to the table
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#d3d3d3")),  # Header background
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),  # Header text color
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # Alignment for all cells
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),  # Header font
        ('FONTSIZE', (0, 0), (-1, 0), 12),  # Header font size
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),  # Header padding
        ('GRID', (0, 0), (-1, -1), 1, colors.black),  # Grid lines
    ])
    table.setStyle(style)

    # Alternate row colors
    for i in range(1, len(table_data)):
        bg_color = colors.whitesmoke if i % 2 == 0 else colors.lightgrey
        ts = TableStyle([('BACKGROUND', (0, i), (-1, i), bg_color)])
        table.setStyle(ts)

    return [table, Spacer(1, 12)]

if __name__ == "__main__":
    stock_ticker = find_ticker()
    if not stock_ticker:
        print("Failed to generate report due to missing ticker symbol.")
        sys.exit(1)
    create_pdf(stock_ticker)

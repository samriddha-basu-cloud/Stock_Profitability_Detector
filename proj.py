from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import plotly.graph_objs as go
from plotly.subplots import make_subplots

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symbol = request.form.get('symbol')
    purchase_date = pd.to_datetime(request.form.get('purchase_date'))

    end_date = purchase_date.strftime('%Y-%m-%d')
    start_date = (purchase_date - pd.DateOffset(years=1)).strftime('%Y-%m-%d')

    data = yf.download(symbol, start=start_date, end=end_date)
    data['Returns'] = data['Adj Close'].pct_change()
    data = data.dropna()

    if data.empty:
        return "No data available for the given date. Please choose a different date."

    X = data[['Open', 'High', 'Low', 'Volume']]
    y = (data['Adj Close'].shift(-1) > data['Adj Close']).astype(int)  # Binary label: 1 if profit, 0 if loss

    model = LogisticRegression()
    model.fit(X, y)

    last_data = X.iloc[-1, :].values.reshape(1, -1)
    prediction = model.predict(last_data)[0]

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['Adj Close'], mode='lines', name='Actual Prices', line=dict(color='blue')))

    fig.update_layout(title=f'Stock Prices for {symbol}', xaxis_title='Date', yaxis_title='Price')

    graph_div = fig.to_html(full_html=False)

    return render_template('result.html', prediction=prediction, graph_div=graph_div)

if __name__ == '__main__':
    app.run(debug=True)


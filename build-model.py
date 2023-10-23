import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from clearml import Task
from alpha_vantage.timeseries import TimeSeries
import os
import matplotlib.pyplot as plt
import joblib
import requests

def fetch_news_sentiment(ticker):
    #API_KEY = "YOUR_ALPHA_VANTAGE_API_KEY"  # Replace with your actual API key
    BASE_URL = "https://www.alphavantage.co/query"
    
    # Make the API request
    response = requests.get(BASE_URL, params={
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "apikey": API_KEY
    })
    
    # Check if the request was successful
    response.raise_for_status()
    
    # Load the JSON data
    data = response.json()

    print(data)
    
    # Extract the sentiment scores and labels for the specified ticker
    sentiment_scores = []
    sentiment_labels = []

    for entry in data['feed']:
        for ticker_data in entry['ticker_sentiment']:
            if ticker_data['ticker'] == ticker:
                sentiment_scores.append(float(ticker_data['ticker_sentiment_score']))
                sentiment_labels.append(ticker_data['ticker_sentiment_label'])
    
    return sentiment_scores, sentiment_labels

def plot_sentiment(task, ticker):
    sentiment_scores, sentiment_labels = fetch_news_sentiment(ticker)
    
    # Create a new figure
    plt.figure(figsize=(14, 7))
    
    # Create a bar plot for sentiment
    bars = plt.bar(range(len(sentiment_scores)), sentiment_scores, color=['green' if s > 0 else 'red' for s in sentiment_scores])
    
    # Annotate bars with sentiment labels
    for bar, label in zip(bars, sentiment_labels):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, label, ha='center', va='bottom', color='black')
    
    # Title, labels, and tweaks
    plt.title(f"News Sentiment for {ticker}")
    plt.ylabel('Sentiment Score')
    plt.xlabel('News Item Index')
    plt.xticks(range(len(sentiment_scores)))
    
    # Display the plot
    plt.tight_layout()
    plt.savefig('sentiment.png')
    task.upload_artifact('sentiment', 'sentiment.png')


def data_preparation(api_key: str, stock: str) -> (Task.id, tuple):
    task = Task.init(project_name='My Project', task_name='Data Preparation')
    
    # Load the training data
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, meta_data = ts.get_daily(symbol=stock, outputsize='full')
    
    data.to_csv('dataset.csv')
    
    # Preprocess the data
    df = pd.read_csv('dataset.csv')
    days = 180
    df = df[::-1]

    # Calculate VWAP
    df['price_avg'] = (df['1. open'] + df['2. high'] + df['3. low'] + df['4. close']) / 4
    df['VWAP'] = (df['price_avg'] * df['5. volume']).cumsum() / df['5. volume'].cumsum()

    data_training = df[df['date']<'2023-01-01'].copy()
    data_training = data_training.drop('date', axis=1)
    scaler = MinMaxScaler(feature_range=(0,1))
    data_training = scaler.fit_transform(data_training)
    
    # Save the scaler for future use
    joblib.dump(scaler, 'scaler.pkl')
    task.upload_artifact('scaler', 'scaler.pkl')

    X_train = []
    y_train = []
    for i, row in enumerate(data_training):
        if i >= days:
            X_train.append(data_training[i-days:i])
            y_train.append(data_training[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    task.upload_artifact('X_train', 'X_train.npy')
    task.upload_artifact('y_train', 'y_train.npy')
    
    task.close()
    return task.id, data_training.shape

def model_training(stock: str, training_data_shape: tuple) -> Task.id:
    task = Task.init(project_name='My Project', task_name=str(stock)+' Training')
    
    # Load preprocessed data
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')

    # Split the training data into training and validation sets
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Load the saved scaler
    scaler = joblib.load('scaler.pkl')

    # Define and train the model
    regressior = Sequential()

    # First LSTM layer
    units_1 = int(os.environ.get('LSTM_UNITS_1', 10))
    regressior.add(LSTM(units=units_1, return_sequences=True, input_shape=(X_train.shape[1], 7)))
    regressior.add(Dropout(0.2))

    # Second LSTM layer
    units_2 = int(os.environ.get('LSTM_UNITS_2', 20))
    regressior.add(LSTM(units=units_2, return_sequences=True))
    regressior.add(Dropout(0.2))

    # Third LSTM layer
    units_3 = int(os.environ.get('LSTM_UNITS_3', 30))
    regressior.add(LSTM(units=units_3))
    regressior.add(Dropout(0.2))

    # Output layer
    regressior.add(Dense(units=1))

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    regressior.compile(optimizer=optimizer, loss='mean_squared_error')

    # Use early stopping to exit training if validation loss is not decreasing after certain epochs
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    # Fit the model
    history = regressior.fit(
        X_train, y_train, 
        epochs=100, 
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping]
    )

    # Save the trained model
    regressior.save(str(stock)+'_model.h5')

    task.upload_artifact(stock+'_model', str(stock)+'_model.h5')
    for epoch, loss in enumerate(history.history['loss']):
        task.get_logger().report_scalar(title='Training Loss', series='Loss', value=loss, iteration=epoch)
    y_pred = regressior.predict(X_train)
    mse = mean_squared_error(y_train, y_pred)
    task.get_logger().report_scalar(title='Prediction Accuracy', series='MSE', value=mse, iteration=epoch)
    
    # Preprocess the data for testing
    df = pd.read_csv('dataset.csv')
    days = 180
    df = df[::-1]

    # Calculate VWAP and price_avg for the test data
    df['price_avg'] = (df['1. open'] + df['2. high'] + df['3. low'] + df['4. close']) / 4
    df['VWAP'] = (df['price_avg'] * df['5. volume']).cumsum() / df['5. volume'].cumsum()

    data_test = df[df['date']>'2023-01-01'].copy()
    data_test_scaled = data_test.drop('date', axis=1)
    data_test_scaled = scaler.transform(data_test_scaled)

    X_test = []
    for i, row in enumerate(data_test_scaled):
        if i >= days:
            X_test.append(data_test_scaled[i-days:i])
    X_test = np.array(X_test)
    
    # Load the model
    model = tf.keras.models.load_model(str(stock)+'_model.h5')

    # Make predictions
    y_pred = model.predict(X_test)
    # Rescale the predictions to the original price scale
    dummy_array = np.zeros(shape=(len(y_pred), training_data_shape[1]))
    dummy_array[:,0] = y_pred[:,0]
    y_pred_original_scale = scaler.inverse_transform(dummy_array)[:,0]

    # Extract actual prices, VWAP, and dates
    actual_prices = df[df['date']>'2023-01-01']['4. close'].values[-len(y_pred_original_scale):]
    actual_vwap = df[df['date']>'2023-01-01']['VWAP'].values[-len(y_pred_original_scale):]
    dates = df[df['date']>'2023-01-01']['date'].values[-len(y_pred_original_scale):]

    # Generate a graph comparing VWAP, Actual Prices, and Predicted Prices
    plt.figure(figsize=(14, 7))
    plt.plot(dates, y_pred_original_scale, label='Predicted Prices', color='blue')
    plt.plot(dates, actual_prices, label='Actual Prices', color='red', linestyle='dashed')
    plt.plot(dates, actual_vwap, label='VWAP', color='green', linestyle='dotted')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('VWAP, Actual and Predicted Prices for ' + stock)
    plt.legend()
    plt.xticks(dates[::10], rotation=45)
    plt.tight_layout()

    # Print the last values under the chart
    last_date = dates[-1]
    last_pred = y_pred_original_scale[-1]
    last_vwap = actual_vwap[-1]
    last_actual = actual_prices[-1]
    info_text = f"Last Date: {last_date}\nPredicted: {last_pred:.2f}\nVWAP: {last_vwap:.2f}\nActual: {last_actual:.2f}"
    plt.text(0.02, 0.02, info_text, transform=plt.gca().transAxes, verticalalignment='bottom', bbox=dict(boxstyle="square,pad=0.3", facecolor="white", edgecolor="black"))

    plt.savefig('vwap_actual_predicted.png')
    task.upload_artifact('vwap_actual_predicted', 'vwap_actual_predicted.png')
    # Check if percentage difference is above a certain threshold
    threshold = 5  # Adjust this value as per your requirement
    # Calculate price difference and percentage difference
    price_difference = y_pred_original_scale - actual_prices
    percentage_difference = (price_difference / actual_prices) * 100
    # Generate a graph of percentage difference
    plt.figure(figsize=(14, 7))
    plt.plot(dates, percentage_difference, label='Percentage Difference', color='green')
    plt.xlabel('Date')
    plt.ylabel('Percentage Difference')
    plt.title('Percentage Difference for ' + stock)
    plt.axhline(0, color='red', linestyle='dashed')
    plt.xticks(dates[::10], rotation=45)
    plt.tight_layout()
    plt.savefig('percentage_difference.png')
    task.upload_artifact('percentage_difference', 'percentage_difference.png')
    plot_sentiment(task,stock)

    if abs(percentage_difference[-1]) > threshold:
        raise ValueError(f"Percentage difference for the last date exceeds {threshold}%!")
    task.close()

if __name__ == "__main__":
    API_KEY = os.environ.get("API_KEY", "changeme")
    stock = os.environ.get("STOCK", "GOOG")
    task_id, training_data_shape = data_preparation(api_key=API_KEY, stock=stock)
    model_training(stock=stock, training_data_shape=training_data_shape)
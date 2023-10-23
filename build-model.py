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
    BASE_URL = "https://www.alphavantage.co/query"
    response = requests.get(BASE_URL, params={
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "apikey": API_KEY
    })
    response.raise_for_status()
    data = response.json()
    print(data)
    sentiment_scores = []
    sentiment_labels = []
    for entry in data['feed']:
        for ticker_data in entry['ticker_sentiment']:
            if ticker_data['ticker'] == ticker:
                sentiment_scores.append(float(ticker_data['ticker_sentiment_score']))
                sentiment_labels.append(ticker_data['ticker_sentiment_label'])
    return sentiment_scores, sentiment_labels

def convert_score_to_category(score):
    if score <= -0.35:
        return "Bearish"
    elif -0.35 < score <= -0.15:
        return "Somewhat-Bearish"
    elif -0.15 < score < 0.15:
        return "Neutral"
    elif 0.15 <= score < 0.35:
        return "Somewhat_Bullish"
    else:
        return "Bullish"

def plot_sentiment(task, ticker):
    sentiment_scores, _ = fetch_news_sentiment(ticker)
    sentiment_categories = [convert_score_to_category(score) for score in sentiment_scores]
    category_counts = dict((category, sentiment_categories.count(category)) for category in set(sentiment_categories))
    plt.figure(figsize=(10, 8))
    plt.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%', startangle=140, colors=['red', 'lightcoral', 'gold', 'yellowgreen', 'green'])
    plt.title(f"News Sentiment Distribution for {ticker}")
    plt.tight_layout()
    plt.savefig('sentiment_pie.png')
    task.upload_artifact('sentiment_pie', 'sentiment_pie.png')
    recommended_sentiment = max(category_counts, key=category_counts.get)
    print(f"The overall recommended sentiment for {ticker} is: {recommended_sentiment}")


def backtest_strategy(predictions, actual_prices):
    """Backtest a simple momentum strategy using predictions."""
    cash = 10000  # Starting cash
    stock_quantity = 0
    initial_cash = cash
    entry_points = []
    exit_points = []
    
    for i in range(len(predictions) - 1):
        # Buy signal: If next day's prediction is higher and we have cash, buy
        if predictions[i + 1] > predictions[i] and cash >= actual_prices[i]:
            stock_quantity += cash // actual_prices[i]
            cash -= stock_quantity * actual_prices[i]
            entry_points.append(i)
        
        # Sell signal: If next day's prediction is lower and we have stock, sell
        if predictions[i + 1] < predictions[i] and stock_quantity > 0:
            cash += stock_quantity * actual_prices[i]
            stock_quantity = 0
            exit_points.append(i)
            
    # If we're holding stock at the end, sell it
    if stock_quantity > 0:
        cash += stock_quantity * actual_prices[-1]
        exit_points.append(len(predictions) - 1)
    
    profit_or_loss = cash - initial_cash
    return entry_points, exit_points, profit_or_loss


def predict_future_days(model, data, scaler, days_in_future):
    predictions = []
    
    # Get the last sequence of data (last 60 days or whatever your sequence length is)
    last_sequence = data[-days_in_future:]
    
    for i in range(days_in_future):
        # Ensure that the last_sequence variable has the correct shape
        last_sequence_reshaped = last_sequence.reshape(1, days_in_future, data.shape[1])
        
        # Predict the next day's price
        next_day_price = model.predict(last_sequence_reshaped)[0][0]
        predictions.append(next_day_price)
        
        # Add the predicted price to the sequence and remove the first element to maintain sequence length
        next_sequence = np.append(last_sequence[1:], [[next_day_price] + list(last_sequence[-1, 1:])], axis=0)
        last_sequence = next_sequence
        
    return predictions


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
    return task.id, data_training.shape, data_training, scaler

def model_training(stock: str, training_data_shape: tuple, data_training, scaler) -> Task.id:
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
    # Second LSTM layer
    units_2 = int(os.environ.get('LSTM_UNITS_2', 20))
    # Third LSTM layer
    units_3 = int(os.environ.get('LSTM_UNITS_3', 30))
    # First LSTM layer
    regressior.add(LSTM(units=units_1, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, input_shape=(X_train.shape[1], 7)))
    # Second LSTM layer
    regressior.add(LSTM(units=units_2, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))
    # Third LSTM layer
    regressior.add(LSTM(units=units_3, dropout=0.4, recurrent_dropout=0.4))

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
        epochs = int(os.environ.get("EPOCHS", 100)),
        batch_size = int(os.environ.get("BATCH_SIZE", 32)),
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

    # List of environment variables of interest
    env_vars = ['LSTM_UNITS_1', 'LSTM_UNITS_2', 'LSTM_UNITS_3', 'EPOCHS', 'BATCH_SIZE']

    # Extract the values of the environment variables
    values = [int(os.environ.get(var, 0)) for var in env_vars]

    # Plotting
    plt.figure(figsize=(10, 6))
    bars = plt.bar(env_vars, values, color='skyblue')
    plt.ylabel('Value')
    plt.title('Environment Variables Visualization')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Displaying the actual values on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, round(yval,2), ha='center', va='bottom')

    plt.savefig('env_data.png')
    task.upload_artifact('env_data', 'env_data.png')

    if abs(percentage_difference[-1]) > threshold:
        raise ValueError(f"Percentage difference for the last date exceeds {threshold}%!")
    # Get the last actual price of the Google stock
    last_actual_price = actual_prices[-1]

    # Calculate future predictions for 7 days using the LSTM model
    future_predictions = predict_future_days(model, data_training, scaler, 7)
    dummy_array = np.zeros(shape=(len(future_predictions), training_data_shape[1]))
    dummy_array[:, 0] = future_predictions
    future_predictions_original_scale = scaler.inverse_transform(dummy_array)[:, 0]

    # Calculate the average increase from the LSTM predictions
    differences = [future_predictions_original_scale[i+1] - future_predictions_original_scale[i] for i in range(len(future_predictions_original_scale)-1)]
    average_increase = sum(differences) / len(differences)

    # Calculate the projected prices based on the average increase for simple projection
    projected_prices_simple = [last_actual_price + i * average_increase for i in range(1, 8)]

    # Plotting both the LSTM predictions and the simple projection
    plt.figure(figsize=(14, 7))

    # Plotting the LSTM model predictions
    plt.plot(range(1, 8), [last_actual_price] + future_predictions_original_scale, 'o-', label='LSTM Predictions', color='blue')

    # Plotting the simple projection based on average increase
    plt.plot(range(8), [last_actual_price] + projected_prices_simple, 's-', label='Simple Projection', color='green')

    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.title('Comparison of LSTM Predictions and Simple Projection for 7 Days')
    plt.legend()
    plt.xticks(range(8))
    plt.tight_layout()
    plt.savefig('lstm_vs_simple_projection.png')
    task.upload_artifact('lstm_vs_simple_projection', 'lstm_vs_simple_projection.png')


    # Backtesting logic starts here
    split_index = int(0.8 * len(y_pred_original_scale))
    backtest_predictions = y_pred_original_scale[split_index:]
    backtest_actual = actual_prices[split_index:]

    entry_points, exit_points, profit_or_loss = backtest_strategy(backtest_predictions, backtest_actual)

    # Predicting the next entry point
    next_entry_point = None
    if len(exit_points) > 0 and exit_points[-1] > entry_points[-1]:  # Last signal was a sell
        for i in range(len(future_predictions_original_scale) - 1):
            if future_predictions_original_scale[i + 1] > future_predictions_original_scale[i]:
                next_entry_point = i + 1
                break

    # Plotting the results
    plt.figure(figsize=(14, 7))
    plt.plot(backtest_actual, label='Actual Prices', color='blue')

    # Plotting the current price
    current_price = backtest_actual[-1]
    plt.axhline(current_price, color='c', linestyle='--', label=f'Current Price: ${current_price:.2f}')

    # Plotting the buy and sell signals
    plt.scatter(entry_points, [backtest_actual[i] for i in entry_points], marker='^', color='g', label='Buy Signal', alpha=1)
    plt.scatter(exit_points, [backtest_actual[i] for i in exit_points], marker='v', color='r', label='Sell Signal', alpha=1)

    # If you have a next entry point predicted, you can add a label for it
    if next_entry_point:
        plt.annotate(f'Next Buy @ Day {next_entry_point}', 
                    xy=(next_entry_point, current_price), 
                    xytext=(next_entry_point+5, current_price+5), 
                    arrowprops=dict(facecolor='black', arrowstyle='->'),
                    horizontalalignment='right')

    plt.title(f'Backtesting Results: Profit/Loss = ${profit_or_loss:.2f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('backtesting_results.png')
    task.upload_artifact('backtesting_results', 'backtesting_results.png')



    task.close()

if __name__ == "__main__":
    API_KEY = os.environ.get("API_KEY", "changeme")
    stock = os.environ.get("STOCK", "GOOG")
    task_id, training_data_shape, data_training, scaler = data_preparation(api_key=API_KEY, stock=stock)
    model_training(stock=stock, training_data_shape=training_data_shape, data_training=data_training, scaler=scaler)
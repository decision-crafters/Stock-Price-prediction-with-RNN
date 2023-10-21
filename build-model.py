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

    # Load the saved scaler
    scaler = joblib.load('scaler.pkl')
    
    # Define and train the model
    regressior = Sequential()
    regressior.add(LSTM(units = 100, activation = 'relu', return_sequences = True, input_shape = (X_train.shape[1], 5), kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    regressior.add(Dropout(0.2))
    regressior.add(LSTM(units = 100, activation = 'relu', return_sequences = True, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    regressior.add(Dropout(0.2))
    regressior.add(LSTM(units = 100, activation = 'relu', return_sequences = True, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    regressior.add(Dropout(0.2))
    regressior.add(LSTM(units = 100, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    regressior.add(Dropout(0.2))
    regressior.add(Dense(units = 1))
    regressior.compile(optimizer='rmsprop', loss='mean_squared_error')
    history = regressior.fit(X_train, y_train, epochs=25, batch_size=64)
    
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

    # Extract actual prices and dates
    actual_prices = df[df['date']>'2023-01-01']['close'].values[-len(y_pred_original_scale):]
    dates = df[df['date']>'2023-01-01']['date'].values[-len(y_pred_original_scale):]

    # Generate a graph of the price prediction
    plt.figure(figsize=(14, 7))
    plt.plot(dates, y_pred_original_scale, label='Predicted Prices', color='blue')
    plt.plot(dates, actual_prices, label='Actual Prices', color='red', linestyle='dashed')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Price Prediction for ' + stock)
    plt.legend()
    plt.xticks(dates[::10], rotation=45)  # Adjust for clarity. Display every 10th date and rotate labels for readability
    plt.tight_layout()
    plt.savefig('price_prediction.png')
    
    task.upload_artifact('price_prediction', 'price_prediction.png')

    
    task.close()

if __name__ == "__main__":
    API_KEY = os.environ.get("API_KEY", "changeme")
    stock = os.environ.get("STOCK", "GOOG")
    task_id, training_data_shape = data_preparation(api_key=API_KEY, stock=stock)
    model_training(stock=stock, training_data_shape=training_data_shape)

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

def data_preparation():
    task = Task.init(project_name='My Project', task_name='Data Preparation')
    
    # Load the training data
    API_KEY = os.environ.get("API_KEY", "changeme")
    stock = os.environ.get("STOCK", "GOOG") 
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    data, meta_data = ts.get_daily(symbol=stock, outputsize='full')  # Unpack the tuple into data and meta_data
    data.to_csv('dataset.csv')

    # Preprocess the data
    df = pd.read_csv('dataset.csv')
    days = 180
    df = df[::-1]
    data_training = df[df['date']<'2021-01-01'].copy()
    data_training = data_training.drop('date', axis=1)
    scaler = MinMaxScaler(feature_range=(0,1))
    data_training = scaler.fit_transform(data_training)

    X_train = []
    y_train = []
    for i in range(days, data_training.shape[0]):
        X_train.append(data_training[i-days:i])
        y_train.append(data_training[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # Save the preprocessed data
    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    task.upload_artifact('X_train', 'X_train.npy')
    task.upload_artifact('y_train', 'y_train.npy')
    task.close()
    return task.id  # Return the task_id from data_preparation

def model_training():
    stock = os.environ.get("STOCK", "GOOG")
    task = Task.init(project_name='My Project', task_name=str(stock)+' Training')
    
    # Load preprocessed data
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    
    # Define and train the model
    regressior = Sequential()
    regressior.add(LSTM(units = 100, activation = 'relu', return_sequences = True, input_shape = (X_train.shape[1], 5)))
    regressior.add(Dropout(0.2))
    regressior.add(LSTM(units = 100, activation = 'relu', return_sequences = True))
    regressior.add(Dropout(0.2))
    regressior.add(LSTM(units = 100, activation = 'relu', return_sequences = True))
    regressior.add(Dropout(0.2))
    regressior.add(LSTM(units = 100, activation = 'relu'))
    regressior.add(Dropout(0.2))
    regressior.add(Dense(units = 1))
    regressior.compile(optimizer='rmsprop', loss='mean_squared_error')
    history = regressior.fit(X_train, y_train, epochs=25, batch_size=64)
    
    # Save the trained model
    regressior.save(str(stock)+'_model.h5')
    task.upload_artifact(stock+'_model', str(stock)+'_model.h5')
    task.close()

def evaluation(data_prep_task_id):
    task = Task.get_task(task_id=data_prep_task_id)
    stock = os.environ.get("STOCK", "GOOG")  # Ensure stock is defined
    
    # Load trained model and test data
    model = tf.keras.models.load_model(task.artifacts[str(stock)+'_model'].get())
    X_train = np.load(task.artifacts['X_train'].get())
    y_train = np.load(task.artifacts['y_train'].get())
    
    # Evaluate the model and generate a report
    train_score = mean_squared_error(y_train, model.predict(X_train))
    loss = train_score  # Assuming loss is the MSE for simplicity
    accuracy = None  # Placeholder, as accuracy isn't provided in the original code
    report = task.create_report()
    report.add_metric('score', train_score)
    report.add_metric('loss', loss)
    report.add_metric('accuracy', accuracy)
    report.send()
    
    task.close()

if __name__ == "__main__":
    data_prep_task_id = data_preparation()  # Capture the task_id from data_preparation
    model_training()  # No need to capture task_id here
    evaluation(data_prep_task_id)  # Pass the task_id from data_preparation to evaluation

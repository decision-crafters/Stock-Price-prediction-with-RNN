from clearml import Task, PipelineDecorator

@PipelineDecorator.component(cache=True, execution_queue="default")
def data_preparation():
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    from alpha_vantage.timeseries import TimeSeries
    import os 

    # Load the training data
    # Load the training data
    API_KEY = os.environ.get("API_KEY", "changeme")
    stock = os.environ.get("STOCK", "GOOG") 
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    data = ts.get_daily(symbol=stock, outputsize='full')
    data.to_csv('dataset.csv')

    # Preprocess the data
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
    
    return X_train, y_train

@PipelineDecorator.component(cache=True, execution_queue="default")
def model_training(X_train, y_train):
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout

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

    # After training, save the model locally
    stock = os.environ.get("STOCK", "GOOG")
    model_path = str(stock)+"_model.h5"
    regressior.save(model_path)

    # Log the saved model to ClearML
    task = Task.current_task()
    task.upload_artifact(name=str(stock)+"model", artifact_object=model_path)
    
    # Log training loss using the correct method
    for epoch, loss in enumerate(history.history['loss']):
        task.get_logger().report_scalar(title='Training Loss', series='Loss', value=loss, iteration=epoch)
    
    return history

@PipelineDecorator.pipeline(
    name='LSTM Training Pipeline',
    project='Stock Price Prediction',
    version='0.1'
)
def pipeline_logic(do_training: bool):
    if do_training:
        X_train, y_train = data_preparation()
        training_history = model_training(X_train, y_train)
        return training_history

if __name__ == '__main__':
    PipelineDecorator.run_locally()
    pipeline_logic(do_training=True)

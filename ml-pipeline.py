from clearml import PipelineDecorator
from clearml import Task

@PipelineDecorator.component(cache=True, execution_queue="default")
def data_preparation():
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np

    # Load the training data
    df = pd.read_csv('dataset.csv')
    
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
    from clearml import Task


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

    # After training, log the model to ClearML
    task = Task.init(project_name='Stock Price Prediction', task_name='LSTM Training')
    model_artifact = task.set_model(model=regressior)

    
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

        # Create a report
        task = Task.current_task()
        report = task.create_report(name="Training Report", type=Task.ReportType.Main)
        report.add_scalar_series(title="Training Loss", series=training_history.history['loss'])
        # Add more metrics or plots as needed
        report.publish()

        return training_history

if __name__ == '__main__':
    PipelineDecorator.run_locally()
    pipeline_logic(do_training=True)

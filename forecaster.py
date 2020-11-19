# Import dependencies
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import keras.backend as K
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout
from scipy import stats

def compileModel(n_steps, n_units, addDropout):
    # Set scope of TensorFlow to global
    K.tensorflow_backend._SYMBOLIC_SCOPE.value = True

    # Reset Tensorflow graph to avoid possible contamination from previous models
    K.clear_session()

    # Instantiate model
    model = Sequential()

    # Apply 1st hidden LSTM layer with input shape (n_steps, number of features) and return sequence output
    model.add(LSTM(units=n_units[0], activation='relu', return_sequences=True, input_shape=(n_steps, 2)))
    if addDropout:
        model.add(Dropout(0.2))

    # Apply 2nd hidden LSTM layer and return sequence output
    model.add(LSTM(units=n_units[1], activation='relu', return_sequences=True))
    if addDropout:
        model.add(Dropout(0.2))

    # Apply 3rd hidden LSTM layer
    model.add(LSTM(units=n_units[2], activation='relu'))
    if addDropout:
        model.add(Dropout(0.2))

    # Apply dense layer to squash output sequence to two vectors
    model.add(Dense(units=2))

    # Compile model with Adam optimizer and MSE for loss calculation
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Store model summary and cast to string
    summary = []
    model.summary(print_fn=lambda x: summary.append(x))

    # Determine index values for deletion
    if addDropout:
        deletionList = [21,20,19,18,17,15,13,11,9,7,5,3,2,1,0]
    else:
        deletionList = [15,14,13,12,11,9,7,5,3,2,1,0]

    # Remove formatting from data
    for index in deletionList:
        del summary[index]

    # Split into columns
    for i in range(len(summary)):
        summary[i] = list(filter(None, summary[i].split('  ')))
        try:
            # Remove empty elements
            summary[i].remove(' ')
        except:
            pass
    
    return model, summary

def preprocessData(df):
    # Slice 'Volume' and 'Adj Close' columns and cast to Numpy array
    data = df.iloc[:, [4,5]].values

    # Normalise data using MinMaxScaler
    scaler = MinMaxScaler((0,1))
    data = scaler.fit_transform(data)

    return data, scaler

def trainModel(df, n_steps, epochs, batch_size, n_units, addDropout):
    # Preprocess data from DataFrame
    data, scaler = preprocessData(df)

    # Declare empty training sets
    X_train = []
    y_train = []

    # Poplulate training sets with data
    for i in range(n_steps, data.shape[0]):
        X_train.append(data[i-n_steps:i, :])
        y_train.append(data[i, :])

    # Cast training sets to Numpy arrays
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Compile model
    model, summary = compileModel(n_steps, n_units, addDropout)
    
    # Train model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    return model, history, summary, data, X_train, scaler

def testModel(model, n_steps, X_train, data):
    # Determine number of elements in test split
    test_split = int(data.shape[0] * 0.3)

    # Populate input test set with data
    X_test = X_train[X_train.shape[0]-test_split-n_steps:]

    # Feed input test set into model to generate ouput test set
    y_test = model.predict(X_test)

    # Isolate only Adj Close values
    y_test = y_test[:,1]

    # Construct list of time data
    times = list(range(y_test.shape[0]))

    # Slice the scaled data to the required length
    actual = list(data[-y_test.shape[0]:,1])
    
    # Construct dictionary of calculated values
    testData = {'actual': actual, 'test': list(y_test), 'times': times}

    return testData

def immediatePrediction(model, data, X_train, n_steps, scaler, todayData):
    # Declare empty input prediction set
    X_pred = []

    # Cast input prediction set to Numpy array
    X_pred = np.array(X_pred)

    # Populate input prediction set with data
    X_pred = data[X_train.shape[0]:, :]
    X_pred = X_pred.reshape(1, n_steps, 2)

    # Predict next data point with model
    prediction = model.predict(X_pred)
    # Inversely transform the predicted data point
    prediction = scaler.inverse_transform(prediction)[0,1]

    # Subtract todays value from yesterday
    diff = prediction - todayData['currentPrice']
        
    # Calculate percentage difference
    pct = (diff / todayData['currentPrice']) * 100

    # Round values to 2.d.p
    diff = round(diff, 2)
    pct = round(pct, 2)

    # Determine whether difference is negative or not
    if diff < 0:
        movement = 'down'
    else:
        movement = 'up'

    # Construct dictionary of calculated values
    immediatePredictionData = {'predictedPrice': round(prediction, 2), 'change': diff, 'percentageChange': pct, 'movement': movement}

    return immediatePredictionData

def generalPrediction(model, data, X_train, n_steps, n_pred):
    # Declare empty prediction sets
    X_pred = []
    y_pred = []

    # Cast prediction sets to Numpy arrays
    X_pred, y_pred = np.array(X_pred), np.array(y_pred)

    # Populate input prediction set with data
    X_pred = data[X_train.shape[0]:, :]
    X_pred = X_pred.reshape(1, n_steps, 2)

    # Create empty array of correct shape to allow appending
    y_pred = np.empty((1, 2))
    y_pred[:] = np.nan

    # Generate forecast values
    for j in range(n_pred):
        prediction = model.predict(X_pred)
        y_pred = np.append(y_pred, prediction, 0)
        X_pred = np.delete(X_pred, [0, 0], 1)
        X_pred = np.append(X_pred[0], prediction, 0)
        X_pred = X_pred.reshape(1, n_steps, 2)
        
    # Remove NaN element
    y_pred = np.delete(y_pred, [0], 0)

    # Create array with values from 0 to n_pred, as x-axis data
    t = range(0, n_pred)
    t = np.asarray(list(t))

    # Calculate slope and product moment correlation coefficient of regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(t, y_pred[:,1])

    # Round values to 4.d.p
    slope = round(slope, 4)
    r_value = round(r_value, 4)

    # Determine whether r value is negative or not
    if r_value < 0:
        movement = 'down'
    else:
        movement = 'up'

    # Construct dictionary of calculated values
    generalPredictionData = {'r_value': r_value, 'slope': slope, 'movement': movement}

    return generalPredictionData
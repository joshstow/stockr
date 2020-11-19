# Import dependencies
from flask import Flask, request, render_template, url_for
import pandas as pd
import scraper
import forecaster as fc
from datetime import datetime

# Instantiate Flask app
app = Flask(__name__)

# Set primary root of app with POST and GET methods
@app.route('/', methods=['POST', 'GET'])
def form_post():
    # Attempt to run operations with form data
    try:
        # Assign form data to variables
        ticker = request.form['ticker']
        n_steps = int(request.form['trainingTimesteps'])
        n_pred = int(request.form['futureTimesteps'])
        date = request.form['date']
        epochs = int(request.form['epochs'])
        batch_size = int(request.form['batchsize'])
        layer1 = int(request.form['layer1'])
        layer2 = int(request.form['layer2'])
        layer3 = int(request.form['layer3'])
        dropout = request.form['dropout']

        # Create tuple with each layer variable
        n_units = (layer1, layer2, layer3)
        
        # Cast dropout variable to Boolean variable addDropout
        if dropout == 'True':
            addDropout = True
        else:
            addDropout = False

        # Construct list of range of epochs
        lossScale = []
        for i in range(epochs):
            lossScale.append(i+1)

        # Construct list of user inputs
        inputs = [('Ticker', ticker.upper()), ('Training timesteps', n_steps), ('Future timesteps', n_pred), ('Start date', date), ('Epochs', epochs), ('Batch size', batch_size), ('LSTM layer units', n_units), ('Dropout', addDropout)]

        # Scrape ticker data for training and store as DataFrame
        df = scraper.Scrape(ticker, date, sel='trainData')
        
        # Get calculations regarding data from today
        todayData = scraper.Today(df)

        # Train model and get history and summary
        model, history, summary, data, X_train, scaler = fc.trainModel(df, n_steps, epochs, batch_size, n_units, addDropout)

        # Test model
        testData = fc.testModel(model, n_steps, X_train, data)

        # Calculate immediate prediction
        immediatePredictionData = fc.immediatePrediction(model, data, X_train, n_steps, scaler, todayData)

        # Calculate general prediction
        generalPredictionData = fc.generalPrediction(model, data, X_train, n_steps, n_pred)

        # Scrape ticker data for charting
        times, adjClose, volume = scraper.Scrape(ticker, date, sel='chartData')

        return render_template('dashboard.html', state='block', inputs=inputs, ticker=ticker.upper(), n_pred=n_pred, currentPrice=todayData['currentPrice'], currentVolume=todayData['currentVolume'], currentChange=todayData['change'], currentPChange=todayData['percentageChange'], currentMovement=todayData['movement'], predictedPrice=immediatePredictionData['predictedPrice'], predictedChange=immediatePredictionData['change'], predictedPChange=immediatePredictionData['percentageChange'], immediatePredictedMovement=immediatePredictionData['movement'], r_value=generalPredictionData['r_value'], slope=generalPredictionData['slope'], generalPredictedMovement=generalPredictionData['movement'], currentTimes=times, adjClose=adjClose, volume=volume, lossScale=lossScale, loss=history.history['loss'], summary=summary, testTimes=testData['times'], actualTestData=testData['actual'], testTestData=testData['test'])

    # Render form only if data from form has not been parsed
    except:
        return render_template('dashboard.html', state='none')

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)

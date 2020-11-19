# Import dependencies
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime

def Scrape(ticker, date, sel):
    # Cast string to datetime object
    start = datetime.strptime(date, "%Y-%m-%d").date()

    # Get current date
    end = datetime.today()

    # Scrape Yahoo Finance for ticker data
    df = web.DataReader(ticker, 'yahoo', start, end)

    # Fill missing days with previous data value
    df = df.asfreq("1D", method="ffill")

    # Determine the way in which the returned data is formatted
    if sel == 'chartData':
        # Convert DataFrame index into list of strings
        times = df.index.tolist()
        for i in range(len(times)):
            times[i] = times[i].strftime("%d-%m-%Y")

        # Convert DataFrame 'Adj Close' column into list of rounded floats
        adjClose = df['Adj Close'].tolist()
        for j in range(len(adjClose)):
            adjClose[j] = round(adjClose[j], 2)

        # Convert DataFrame 'Volume' column into list
        volume = df['Volume'].tolist()

        return times, adjClose, volume

    elif sel == 'trainData':
        return df

def Today(df):
    # Isolate last two columns from DataFrame
    df = df.tail(2)

    # Subtract todays value from yesterday
    diff = df['Adj Close'][1] - df['Adj Close'][0]

    # Calculate percentage difference
    pct = (diff / df['Adj Close'][0]) * 100

    # Round values to 2.d.p
    diff = round(diff, 2)
    pct = round(pct, 2)

    # Determine whether difference is negative or not
    if diff < 0:
        movement = 'down'
    else:
        movement = 'up'

    # Convert volume data into integer values and add commas to seperate thousands
    volume = int(df['Volume'][1])
    volume = '{:,}'.format(volume)

    # Construct dictionary of calculated values
    todayData = {'currentPrice': round(df['Adj Close'][1], 2), 'currentVolume': volume, 'change': diff, 'percentageChange': pct, 'movement': movement}

    return todayData
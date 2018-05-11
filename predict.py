#!/usr/bin/python
import csv
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

#train and test dataset variables*
dates = list()
prices = list()
x_train = x_test = y_train = y_test = list()
split=int()

def get_data(filename):
        """Import the dataset into a dataframe and separate train and test dataset. x is for dates and y for prices. Dataframe has 2 columns Date and Open rice"""
        global split, x_train, x_test, y_train, y_test, dates, prices

        df = pd.read_csv(filename,parse_dates=['Date'])
        df = df[['Date','Open']]
        df = df.sort_values(by='Date')
        split = int(len(df)*0.8) #use 80% dataset for training and 20% for testing

        with open(filename,'r') as csvfile:
                csvFileReader = csv.reader(csvfile)
                next(csvFileReader)
                for row in csvFileReader:
                        dates.append(int(row[0].split('-')[2]))
                        prices.append(float(row[1]))

        x_train, x_test, y_train, y_test = dates[0:split], dates[split:len(dates)], prices[0:split], prices[split:len(prices)]

        return df

def predict_price(x,y,d,interval):
    """Create Reression model and use it to predict prices."""
    linear_mod = linear_model.LinearRegression() #Linear Regression Model
    x = np.reshape(x,(len(x),1))
    y = np.reshape(y,(len(y),1))

    svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1) #RBF Model
    svr_rbf.fit(x, y.ravel()) #.ravel() create flattened 1D array for RBF model
    linear_mod.fit(x, y)

    predicted_price = list()
    y = int()
    date = pd.date_range(d, periods=interval+1).tolist() #creates list of date after the last date of dataset

    for y in range(0,interval+1):
            date[y] = int(date[y].strftime('%Y-%m-%d').split('-')[2])

    date=np.reshape(date,(len(date),1))
    predicted_price = pd.DataFrame(data={'RBF':svr_rbf.predict(date)})
    predicted_price['Linear'] = linear_mod.predict(date)

    return predicted_price,linear_mod.coef_[0][0] ,linear_mod.intercept_[0]

def dataframe(df,interval,predicted_price):
        """Gather all the predicted values and dates into single dataframe."""
        y = df['Date'][split]
        datelist = pd.date_range(y, periods=interval+1).tolist()

        for y in range(0,interval+1):
                datelist[y] = datelist[y].strftime('%Y-%m-%d')

        dl = pd.DataFrame(data={'Date':datelist})
        dl['RBF'] = predicted_price['RBF'].values
        dl['Linear'] = predicted_price['Linear'].values
        dl['Date'] = pd.to_datetime(dl['Date']).values

        return dl[len(x_test):]

def testing(predicted_price):
        """Test the predicted prices with test dataset."""
        global y_test
        test1 = list(range(0,len(y_test)))
        test2 = list(range(0,len(y_test)))
        a=int()

        for a in range(0,len(y_test)):
               test1[a] = y_test[a] - predicted_price['Linear'][a]
               predicted_price['Linear'][a] = predicted_price['Linear'][a] + test1[a]
               test2[a] = y_test[a] - predicted_price['RBF'][a]
               predicted_price['RBF'][a] = predicted_price['RBF'][a] + test2[a]

        avr1 = sum(test1)/len(test1)
        avr2 = sum(test2)/len(test2)

        for a in range(len(y_test),len(predicted_price['Linear'])):
                predicted_price['Linear'][a] = predicted_price['Linear'][a] + avr1
                predicted_price['RBF'][a] = predicted_price['RBF'][a] + avr2

        return int(avr1), int(avr2)

def clear():
	global split, x_train, x_test, y_train, y_test, dates, prices
	split = int()
	dates = list()
	prices = list()
	x_train = x_test = y_train = y_test = list()
	return

def main(inter,filename):
    df = get_data(filename)
    date = df['Date'][split]
    interval = int(inter) + len(x_test)

    predicted_price, coefficient, constant = predict_price(x_train, y_train, date, interval)
    avr1, avr2 = testing(predicted_price)
    predicted = dataframe(df, interval, predicted_price)
    clear()

    if __name__ = '__main__':
        print ('The stock open prices for following dates are:')
        print (predicted)
        print ('The regression coefficient is ' + str(coefficient) + ', the constant is ' +  str(constant) + ', and the error is ' + str(avr1/100) + '%')
        print ('The relationship equation between dates and prices is: price = ' + str(coefficient) + '* date + ' + str(constant))

    dates = predicted['Date'].dt.date.values

    return [predicted['Linear'].values, predicted['RBF'].values, dates, avr1, avr2, coefficient, constant]

ls = main(40,'dataset.csv')

#Data from S&P 500 index
#Indexes aggregate prices of mulitple stocks put together
#...allow you to see how market as a whole is doing

#ETF (Exchange Trade Fund)
#..allow you to buy and sell indexes like stocks

#Each row in data contains daily record of S&P 500 price
#..from 1950 to 2015

#Train model with data from 1950-2012 to predict 2013-15
import pandas as pd
from datetime import datetime

data = pd.read_csv('sphist.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by='Date')
#print(len(data))


#Stock market data isn't independent
#..each day's data is somewhat dependant upon prior day

#Time series nature means you can generate indicators to market
#model more accurate

#Want to generate columns for 3 "indicators"
#..avg of prior 5 days, month, etc...

#Average price past xx days
def compute_avg_x_days(n_days):
    n_day_avg = []
    for index in range(len(data)):
        if index <= (n_days - 1):
            n_day_avg.append(0)
        else:
            past_x_days = data.iloc[(index - n_days) : index]
            avg_close = past_x_days['Close'].mean()
            n_day_avg.append(avg_close)
    new_column = 'avg_' + str(n_days) + '_days'
    data[new_column] = pd.Series(n_day_avg).values

#Standard Deviation of past xx days
def std_x_days(n_days):
    n_day_avg = []
    for index in range(len(data)):
        if index <= (n_days - 1):
            n_day_avg.append(0)
        else:
            past_x_days = data.iloc[(index - n_days) : index]
            avg_close = past_x_days['Close'].std()
            n_day_avg.append(avg_close)
    new_column = 'stdev_' + str(n_days) + '_days'
    data[new_column] = pd.Series(n_day_avg).values

compute_avg_x_days(5)
compute_avg_x_days(365)
std_x_days(5)
std_x_days(365)

#Test to confirm anwers are correct
#print(data.loc[(data['Date'] >= '1960-12-26') & (data['Date'] <= '1961-01-04')])

data = data[(data != 0).all(1)]
print(data.head())

#Make DF for Train (Before 2013) and Test (After 2013)
train = data[data['Date'] < '2013-01-01']
test = data[data['Date'] >= '2013-01-01']

#Confirm we're not missing any rows
print('train: ' + str(train.shape[0]))
print('test: ' + str(test.shape[0]))
print('full: ' + str(data.shape[0]))

#Initialize instance of LR class
#Train...leave out all original columns
#Use Close as your target
#Make predictions for Close using same columns for training as you did with train
#Compute error between predictions and Close column of test
#Use MAE as error metric
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

features = list(data.columns)
features = [i for i in features if i not in ('Close', 'High', 'Low', 'Open', 'Volume',
                                             'Adj Close', 'Date')]
#features.remove('Close').remove('Date')
#print(features)
lr = LinearRegression()
lr.fit(train[features], train['Close'])
predictions = lr.predict(test[features])
print(predictions)
test['predictions'] = predictions
print(test[['Close', 'predictions']].head())

mae = mean_absolute_error(test['Close'], test['predictions'])
print('MAE: ' + str(mae))


#Make graph showing how predictions compare to actual values

#...

#How to improve Error
#...
#1. Avg volumne over 5 days / year?
#2. Std dev of average volumne over past 5 days / year?
#3. Year component of date
#4. Day of week
#5. Number of holidays in prior month
#6. Ratio between lowest price in past year and current price in past year

#Add these additional indicators to the dataframe
#...must insert these at the same point where you insert others, before clearing out NaN (0) valeus


#Other ways to improve algorithm
#1. Make predictions only one day ahead
###Train(1/3/51 --> 1/2/13) to predict (1/3/13)
###...then (1/3/51 --> 1/3/13) to predict (1/4/13)

#2. Random Forest?
#3. Incorporate other data (like weather, Twitter activity around stocks, etc.)
#4. Automate script to download latest data to make predictions for next day
#5. You can make hourly / minute-by-minute / second-by-second predictions

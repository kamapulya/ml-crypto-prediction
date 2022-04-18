# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 23:45:07 2022

@author: Stas
"""
import pandas as pd
from binance.client import Client
import datetime as dt

# client configuration
api_key = 'YOUR API KEY' 
api_secret = 'YOUR SECRET KEY'
client = Client(api_key, api_secret)

def get_gistorical_prices(symbol, interval, startdate, enddate):
    klines = client.get_historical_klines(symbol, interval, startdate, enddate)
    df = pd.DataFrame(klines)
    df.columns = ['open_time','open', 'high', 'low', 'close', 'volume','close_time','qav','num_trades','taker_base_vol','taker_quote_vol', 'ignore']
    df.index = [dt.datetime.fromtimestamp(x/1000.0) for x in df.close_time]
    
    return df

symbol = "ETHUSDT"
start_date = "1 Oct, 2021"
stop_date = "30 Oct, 2021"
interval = Client.KLINE_INTERVAL_1MINUTE

#fetching the data
df = get_gistorical_prices(symbol, Client.KLINE_INTERVAL_1MINUTE, start_date, stop_date)

#convert to csv
df.to_csv(symbol+'.csv', index = None, header=True)

#convert data to float and plot
df = df.astype(float)
df["close"].plot(title = 'symbol', legend = 'close')

#symbol = "BTCUSDT"
#start_date = "1 Oct, 2021"
#stop_date = "30 Oct, 2021"

#klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1MINUTE, start_date, stop_date)

# fetch 1 minute klines for the last day up until now
#klines = client.get_historical_klines("BNBBTC", Client.KLINE_INTERVAL_1MINUTE, "1 day ago UTC")

# fetch 30 minute klines for the last month of 2017
#klines = client.get_historical_klines("ETHBTC", Client.KLINE_INTERVAL_30MINUTE, "1 Dec, 2017", "1 Jan, 2018")

# fetch weekly klines since it listed
#klines = client.get_historical_klines("NEOBTC", Client.KLINE_INTERVAL_1WEEK, "1 Jan, 2017")

#data = pd.DataFrame(klines)
 # create colums name
#data.columns = ['open_time','open', 'high', 'low', 'close', 'volume','close_time', 
#                'qav','num_trades','taker_base_vol','taker_quote_vol', 'ignore']
            
# change the timestamp
#data.index = [dt.datetime.fromtimestamp(x/1000.0) for x in data.close_time]
#data.to_csv(symbol+'.csv', index = None, header=True)

#convert data to float and plot
#data = data.astype(float)
#data["close"].plot(title = symbol, legend = 'close')


"""
def get_klines_iter(symbol, interval, start, end, limit=5000):
    df = pd.DataFrame()
    startDate = end
    while startDate>start:
        url = 'https://api.binance.com/api/v3/klines?symbol=' + \
            symbol + '&interval=' + interval + '&limit='  + str(iteration)
        if startDate is not None:
            url += '&endTime=' + str(startDate)
        
        df2 = pd.read_json(url)
        df2.columns = ['Opentime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Closetime', 'Quote asset volume', 'Number of trades','Taker by base', 'Taker buy quote', 'Ignore']
        df = pd.concat([df2, df], axis=0, ignore_index=True, keys=None)
        startDate = df.Opentime[0]   
    df.reset_index(drop=True, inplace=True)    
    return df 
"""
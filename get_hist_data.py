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
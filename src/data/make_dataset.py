import json
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from datetime import timedelta
from pandas import json_normalize

def get_data(symbol='BTC_JPY', interval='1hour', date=''):
    endPoint = 'https://api.coin.z.com/public'
    path     = f'/v1/klines?symbol={symbol}&interval={interval}&date={date}'

    response = requests.get(endPoint + path)
    r = json.dumps(response.json(), indent=2)
    r2 = json.loads(r)
    df = json_normalize(r2['data'])
    if len(df):
        date = []
        for i in df['openTime']:
            i = int(i)
            tsdate = int (i / 1000)
            loc = datetime.utcfromtimestamp(tsdate)
            date.append(loc)
        df.index = date
        df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('Asia/Tokyo')
        df.drop('openTime', axis=1, inplace=True)
    return df

def get_today():
    now_time = datetime.now()
    current_year = now_time.year
    current_month = now_time.month
    current_day = now_time.day
    if current_month >= 10 and current_day >= 10:
        today = str(current_year) + str(current_month) + str(current_day)
    elif current_month < 10 and current_day >= 10:
        today = str(current_year) + '0' + str(current_month) + str(current_day)
    elif current_month < 10 and current_day < 10:
        today = str(current_year) + '0' + str(current_month) + '0' + str(current_day)
    elif current_month >= 10 and current_day < 10:
        today = str(current_year) + str(current_month) + '0' + str(current_day)
    return today

def get_train_data():
    '''データ取得'''
    today = get_today()
    day = datetime.strptime(today, '%Y%m%d')
    day -= timedelta(days=1)
    day = str(day)
    day = day.replace('-', '')
    day = day.replace(' 00:00:00', '')
    btc_today = get_data(symbol='BTC_JPY', interval='1hour', date=day)
    eth_today = get_data(symbol='ETH', interval='1hour', date=day)
    if len(eth_today) == 0:
        eth_today = pd.DataFrame(data=np.array([[None for i in range(24)] for i in range(5)]).T, index=btc_today.index, columns=['eth_open', 'eth_high', 'eth_low', 'eth_close', 'eth_volume'])
    eth_today.columns = ['eth_open', 'eth_high', 'eth_low', 'eth_close', 'eth_volume']
    df = pd.concat([btc_today, eth_today], axis=1)
    for i in range(480):
        day = datetime.strptime(day, '%Y%m%d')
        day -= timedelta(days=1)
        day = str(day)
        day = day.replace('-', '')
        day = day.replace(' 00:00:00', '')
        btc = get_data(symbol='BTC_JPY', interval='1hour', date=day)
        eth = get_data(symbol='ETH', interval='1hour', date=day)
        if len(eth) == 0:
            eth = pd.DataFrame(data=np.array([[None for i in range(24)] for i in range(5)]).T, index=btc.index, columns=['eth_open', 'eth_high', 'eth_low', 'eth_close', 'eth_volume'])
        eth.columns = ['eth_open', 'eth_high', 'eth_low', 'eth_close', 'eth_volume']
        tmp_df = pd.concat([btc, eth], axis=1)
        df = pd.concat([tmp_df, df], axis=0)
    return df

def data_for_prediction():
    today = get_today()
    yesterday = datetime.strptime(today, '%Y%m%d')
    yesterday -= timedelta(days=1)
    yesterday = str(yesterday)
    yesterday = yesterday.replace('-', '')
    yesterday = yesterday.replace(' 00:00:00', '')
    two_days_before = datetime.strptime(yesterday, '%Y%m%d')
    two_days_before -= timedelta(days=1)
    two_days_before = str(two_days_before)
    two_days_before = two_days_before.replace('-', '')
    two_days_before = two_days_before.replace(' 00:00:00', '')
    if datetime.now().hour > 6:
        btc_today = get_data(symbol='BTC_JPY', interval='1hour', date=today)
        eth_today = get_data(symbol='ETH', interval='1hour', date=today)
        btc_yesterday = get_data(symbol='BTC_JPY', interval='1hour', date=yesterday)
        eth_yesterday = get_data(symbol='ETH', interval='1hour', date=yesterday)
        btc_data = pd.concat([btc_yesterday, btc_today], axis=0)
        eth_data = pd.concat([eth_yesterday, eth_today], axis=0)
        eth_data.columns = ['eth_open', 'eth_high', 'eth_low', 'eth_close', 'eth_volume']
        df = pd.concat([btc_data, eth_data], axis=1)
        df = df.tail(20)
    else:
        btc_yesterday = get_data(symbol='BTC_JPY', interval='1hour', date=yesterday)
        eth_yesterday = get_data(symbol='ETH', interval='1hour', date=yesterday)
        btc_day2 = get_data(symbol='BTC_JPY', interval='1hour', date=two_days_before)
        eth_day2 = get_data(symbol='ETH', interval='1hour', date=two_days_before)
        btc_data = pd.concat([btc_day2, btc_yesterday], axis=0)
        eth_data = pd.concat([eth_day2, eth_yesterday], axis=0)
        eth_data.columns = ['eth_open', 'eth_high', 'eth_low', 'eth_close', 'eth_volume']
        df = pd.concat([btc_data, eth_data], axis=1)
        df = df.tail(20)
    return df
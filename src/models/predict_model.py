import json
import numpy as np
import requests
from datetime import datetime
from pandas import json_normalize


def normalise_windows(window_data, single_window=False):
    	#''' window normalization'''
    normalised_data = [] # 正規化したデータを格納
    window_data = [window_data] if single_window else window_data
    for window in window_data:
        normalised_window = []
        for col_i in range(window.shape[1]): # Windowの幅
            # 各値を初期の値で割る
            normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
            normalised_window.append(normalised_col)
            # reshape and transpose array back into original multidimensional format
        normalised_window = np.array(normalised_window).T
        normalised_data.append(normalised_window)
    return np.array(normalised_data)

def get_data(symbol='BTC', interval='1hour', date=''):
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




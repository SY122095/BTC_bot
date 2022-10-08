import configparser
import hashlib
import hmac
import json
import requests
import time
import pandas as pd
from pytz import timezone
from datetime import datetime
from line.line_notify import LineNotify

line_notify = LineNotify()


# APIキーの設定
conf = configparser.ConfigParser()
conf.read('config.ini')
api_Key = conf['gmo']['api_Key']
secretKey = conf['gmo']['secretKey']

#------------------------GMOコインAPIを用いた取引目的の関数------------------------#
def get_price(symbol='BTC_JPY'):
    '''
    仮想通貨の現在価格を取得する関数
    params
    ============
    symbol: str
        取得する仮想通貨名
    '''
    endPoint = 'https://api.coin.z.com/public'
    path     = f'/v1/ticker?symbol={symbol}'

    response = requests.get(endPoint + path)
    r = json.dumps(response.json(), indent=2)
    return json.loads(r)['data'][0]['ask']
    
def get_availableAmount():
    '''
    取引余力を取得する関数
    '''
    timestamp = '{0}000'.format(int(time.mktime(datetime.now().timetuple())))
    method    = 'GET'
    endPoint  = 'https://api.coin.z.com/private'
    path      = '/v1/account/margin'
    text = timestamp + method + path
    sign = hmac.new(bytes(secretKey.encode('ascii')), bytes(text.encode('ascii')), hashlib.sha256).hexdigest()
    
    headers = {
        "API-KEY": api_Key,
        "API-TIMESTAMP": timestamp,
        "API-SIGN": sign
    }
    
    res = requests.get(endPoint + path, headers=headers)
    r =  json.dumps(res.json(), indent=2)
    return json.loads(r)['data']['availableAmount']
    

def build_position(symbol, side, executionType, size, price='', losscutPrice='', timeInForce='FAK'):
    '''
    ポジションを決める
    prameters
    =============
    symbol: str
        注文する銘柄
    executionType: MARKET LIMIT STOP
        成行、指値、逆指値
    timeInForce: 
    price: int, float
        注文価格, 指値の場合は必須
    losscutPrice: 
    size: int, float
        注文数量
    '''
    timestamp = '{0}000'.format(int(time.mktime(datetime.now().timetuple())))
    method    = 'POST'
    endPoint  = 'https://api.coin.z.com/private'
    path      = '/v1/order'
    reqBody = {
        "symbol": symbol,
        "side": side,
        "executionType": executionType,
        "timeInForce": timeInForce,
        "price": price,
        "losscutPrice": losscutPrice,
        "size": size
    }

    text = timestamp + method + path + json.dumps(reqBody)
    sign = hmac.new(bytes(secretKey.encode('ascii')), bytes(text.encode('ascii')), hashlib.sha256).hexdigest()

    headers = {
        "API-KEY": api_Key,
        "API-TIMESTAMP": timestamp,
        "API-SIGN": sign
    }

    res = requests.post(endPoint + path, headers=headers, data=json.dumps(reqBody))
    return res.json()
    

def get_position():
    '''建玉一覧を取得'''
    timestamp = '{0}000'.format(int(time.mktime(datetime.now().timetuple())))
    method    = 'GET'
    endPoint  = 'https://api.coin.z.com/private'
    path      = '/v1/openPositions'

    text = timestamp + method + path
    sign = hmac.new(bytes(secretKey.encode('ascii')), bytes(text.encode('ascii')), hashlib.sha256).hexdigest()
    parameters = {
        "symbol": "BTC_JPY",
        "page": 1,
        "count": 100
    }

    headers = {
        "API-KEY": api_Key,
        "API-TIMESTAMP": timestamp,
        "API-SIGN": sign
    }

    res = requests.get(endPoint + path, headers=headers, params=parameters)
    return res.json()

def close_position(ticker, side, size, executionType, position_id):
    '''決済注文を出す'''
    timestamp = '{0}000'.format(int(time.mktime(datetime.now().timetuple())))
    method    = 'POST'
    endPoint  = 'https://api.coin.z.com/private'
    path      = '/v1/closeOrder'
    reqBody = {
        "symbol": ticker,
        "side": side,
        "executionType": executionType,
        "timeInForce": "",
        "price": "",
        "settlePosition": [
            {
                "positionId": position_id,
                "size": size
            }
        ]
    }

    text = timestamp + method + path + json.dumps(reqBody)
    sign = hmac.new(bytes(secretKey.encode('ascii')), bytes(text.encode('ascii')), hashlib.sha256).hexdigest()

    headers = {
        "API-KEY": api_Key,
        "API-TIMESTAMP": timestamp,
        "API-SIGN": sign
    }

    res = requests.post(endPoint + path, headers=headers, data=json.dumps(reqBody))
    return res.json()
    
    
def exe_position(side, position):
    '''ポジションの決済を実行'''
    if side == 'BUY':
        close_side = 'SELL'
    elif side == 'SELL':
        close_side = 'BUY'
    
    for i in position:
        if i['side'] == 'BUY':
            close_res = close_position(i['symbol'], close_side, i['size'], 'MARKET', i['positionId'])
            if close_res['status'] == 0:
                print('レバレッジ取引(買い注文)は決済されました')
            else:
                print(close_res)
        elif i['side'] == 'SELL':
            close_res = close_position(i['symbol'], close_side, i['size'], 'MARKET', i['positionId'])
            if close_res['status'] == 0:
                print('レバレッジ取引(売り注文)は決済されました')
            else:
                print(close_res)
                

def exe_all_position():
    '''すべてのポジションを決済する'''
    position = get_position()
    if position['data'] == {}:
        print('ポジションはありません')
    else:
        for i in position['data']['list']:
            if i['side'] == 'BUY':
                close_res = close_position(i['symbol'], 'SELL', i['size'], 'MARKET', i['positionId'])
                if close_res['status'] == 0:
                    print('レバレッジ取引(買い注文)は決済されました')
                    line_notify.send('レバレッジ取引(買い注文)は決済されました')
                else:
                    print(close_res)
            elif i['side'] == 'SELL':
                close_res = close_position(i['symbol'], 'BUY', i['size'], 'MARKET', i['positionId'])
                if close_res['status'] == 0:
                    print('レバレッジ取引(売り注文)は決済されました')
                    line_notify.send('レバレッジ取引(買い注文)は決済されました')
                else:
                    print(close_res)
                    
                    
def order_process(symbol, side, executionType, size, price='', losscutPrice='', timeInForce='FAK'):
    '''注文を出す'''
    if side == 'BUY':
        build_position(symbol, side, executionType, size, price, losscutPrice, timeInForce='FAK')
        time.sleep(1)
        print('ビットコインを' + str(get_position()['data']['list'][0]['price']) + '円でロングしました')
        line_notify.send('ビットコインを' + str(get_position()['data']['list'][0]['price']) + '円でロングしました')
    elif side == 'SELL':
        build_position(symbol, side, executionType, size, price, losscutPrice, timeInForce='FAK')
        time.sleep(1)
        print('ビットコインを' + str(get_position()['data']['list'][0]['price']) + '円でショートしました')
        line_notify.send('ビットコインを' + str(get_position()['data']['list'][0]['price']) + '円でショートしました')
        

def get_trading_info():
    '''最新の約定情報を取得'''
    timestamp = '{0}000'.format(int(time.mktime(datetime.now().timetuple())))
    method    = 'GET'
    endPoint  = 'https://api.coin.z.com/private'
    path      = '/v1/latestExecutions'
    
    text = timestamp + method + path
    sign = hmac.new(bytes(secretKey.encode('ascii')), bytes(text.encode('ascii')), hashlib.sha256).hexdigest()
    parameters = {
        "symbol": "BTC",
        "page": 1,
        "count": 1
    }

    headers = {
        "API-KEY": api_Key,
        "API-TIMESTAMP": timestamp,
        "API-SIGN": sign
    }

    res = requests.get(endPoint + path, headers=headers, params=parameters)
    r =  json.dumps(res.json(), indent=2)
    #return json.loads(r)
    return json.loads(r)['data']['list'][0]


def get_trade_result():
    '''取引の記録を取得'''
    timestamp = '{0}000'.format(int(time.mktime(datetime.now().timetuple())))
    method    = 'GET'
    endPoint  = 'https://api.coin.z.com/private'
    path      = '/v1/latestExecutions'

    text = timestamp + method + path
    sign = hmac.new(bytes(secretKey.encode('ascii')), bytes(text.encode('ascii')), hashlib.sha256).hexdigest()
    parameters = {
        "symbol": "BTC_JPY",
        "page": 1,
        "count": 2
    }

    headers = {
        "API-KEY": api_Key,
        "API-TIMESTAMP": timestamp,
        "API-SIGN": sign
    }

    res = requests.get(endPoint + path, headers=headers, params=parameters)
    time_ = res.json()['data']['list'][1]['timestamp']
    time_ = pd.Timestamp(time_)
    time_ = time_.astimezone(timezone('Asia/Tokyo'))
    year = time_.year
    month = time_.month
    day = time_.day
    hour = time_.hour
    date = str(year) + '-' + str(month)+ '-' + str(day)+ ' ' + str(hour) + ':00:00'
    side = res.json()['data']['list'][1]['side']
    if side == 'SELL':
        position = -1
    else:
        position = 1
    order_price = int(res.json()['data']['list'][1]['price'])
    close_price = int(res.json()['data']['list'][0]['price'])
    loss_gain = int(res.json()['data']['list'][0]['lossGain'])
    id = int(res.json()['data']['list'][0]['executionId'])
    
    return id, date, position, order_price, close_price, loss_gain
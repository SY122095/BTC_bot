import pandas as pd
import tensorflow as tf
import time
import numpy as np
import sqlite3
from src.data.make_dataset import data_for_prediction, get_train_data
from src.features.build_features import data_cleaning, feature_engineering, create_x_for_prediction
from src.models.predict_model import get_today
from src.models.train_model import wavenet_training, display_result, mish
from trading.trading import get_price, get_availableAmount, exe_all_position, order_process, get_trade_result
from line.line_notify import LineNotify
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


####-----------------------------初期設定-----------------------------####
day = get_today() # 運用開始日の日付(YYYYMMDD形式)
hour = datetime.now().hour # 運用開始時間
ticker = "BTC_JPY" # 売買対象の仮想通貨(ビットコインのレバレッジ取引)
exe_type = 'MARKET' # 注文方式(成行)
default_balance = float(get_availableAmount()) # デフォルトの残高
line_notify = LineNotify() # 取引発生時にLINEで知らせるためのインスタンス
line_notify.send('取引ボットの稼働を開始します。')
result_df = pd.DataFrame(columns=['id', 'date', 'position', 'order_price', 'close_price', 'loss_gain']) # 取引結果を格納するデータフレーム
trade_num = 0 # 取引回数
df_num = 1
profit = 0
#get_custom_objects().update({'mish': mish})
tf.keras.utils.get_custom_objects().update({'mish': mish})
dbname = 'sql/trading.db' # 取引結果を格納するテーブル


####-----------------------------Bot本体の処理-----------------------------####
while True:
    # apiのメンテナンス時はスキップ
    
    # 1時間経過したら取引を行う
    if hour != datetime.now().hour:
        try:
            price = get_price()
            print(f'現在の{ticker}価格は{price}円です。\n')
        except:
            print('メンテナンス中です。\n')
            line_notify.send('メンテナンス中です。')
            hour = datetime.now().hour
            continue
        print(datetime.now())
        print('予測を行います。')
        exe_all_position() # ポジションを決済
        ##--------ポジションを決めるための予測を行う--------##
        df = data_for_prediction()
        x = create_x_for_prediction(df)
        model = tf.keras.models.load_model('.\\models\\model.h5', custom_objects={'mish': mish})
        prediction = model.predict(x)
        print(prediction)
        if prediction > 0:
            side = 'BUY'
            tmp_position = 'long'
            hour = datetime.now().hour
        elif prediction <= 0:
            side = 'SELL'
            tmp_position = 'short'
            hour = datetime.now().hour
        else:
            continue
        print(str(datetime.now().year) + '年' + str(datetime.now().month) + '月' + str(datetime.now().day) + '日' + str(datetime.now().hour) + '時のポジションは' + tmp_position + 'です。')
        time.sleep(1)
        if trade_num != 0:
            tmp_id, tmp_date, tmp_position_, tmp_order_price, tmp_close_price, tmp_loss_gain = get_trade_result() # 取引記録の取得
            tmp_df = pd.DataFrame(columns=['id', 'date', 'position', 'order_price', 'close_price', 'loss_gain'],
                        data=[[tmp_id, tmp_date, tmp_position_, tmp_order_price, tmp_close_price, tmp_loss_gain]])
            result_df = pd.concat([result_df, tmp_df])
            profit += tmp_loss_gain
            # データベースに登録
            conn = sqlite3.connect(dbname)
            cur = conn.cursor()
            cur.execute('INSERT INTO trading values(?, ?, ?, ?, ?, ?)', (tmp_id, tmp_date, tmp_position_, tmp_order_price, tmp_close_price, tmp_loss_gain))
            conn.commit()
            conn.close()
            print('取引結果をデータベースに登録しました。\n')
        
        available = int(get_availableAmount())
        profit_loss = available - default_balance
        profit_rate = profit / default_balance
        if profit_loss > 0:
            print('現在の残高は' + str(available) + '円で、' + str(profit_loss) + '円の利益です\n')
            line_notify.send('現在の残高は' + str(available) + '円で、' + str(profit_loss) + '円の利益です')
        elif profit_loss == 0:
            print('現在の残高は' + str(available) + '円で、' + '損益無しです\n')
            line_notify.send('現在の残高は' + str(available) + '円で、' + '損益無しです')
        else:
            print('現在の残高は' + str(available) + '円で、' + str(-profit_loss) + '円の損失です\n')
            line_notify.send('現在の残高は' + str(available) + '円で、' + str(-profit_loss) + '円の損失です')
        if profit_rate < -0.2:
            break
        ##--------注文を出す--------##
        order_process(symbol=ticker, side=side, executionType=exe_type, size=0.01)
        time.sleep(3)
        trade_num += 1
            
        ##--------モデルの更新--------##
        tmp_day = get_today()
        if tmp_day != day:
            day = tmp_day
            if len(result_df) != 0:
                #result_df['direction'] = result_df['position'].apply(lambda x: x if result_df['loss_gain']>=0 else -x)
                result_df['direction'] = 1
                for i in range(len(result_df)):
                    if result_df['loss_gain'][i] < 0:
                        result_df['direction'][i] == -1
                result_df['return'] = np.log(result_df['close_price'] / result_df['order_price'])
                result_df['strategy'] = result_df['return'] * result_df['position']
                benchmark = result_df['return'].sum()
                wavenet_return = result_df['strategy'].sum()
                accuracy = accuracy_score(result_df['direction'], result_df['position'])
                if accuracy < 0.5 and wavenet_return < benchmark:
                    df = get_train_data()
                    df = data_cleaning(df)
                    x, y = feature_engineering(df)
                    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, shuffle=False)
                    model = wavenet_training(x_train, x_valid, y_train, y_valid, num_filters=16, kernel_size=8, batchsize=128, lr=0.0001)
                    display_result(model, x_valid, y_valid, df)
                    tf.keras.models.save_model(model, '.\\models\\model.h5')
                    result_df.to_csv(f'.\\results\\result_{df_num}.csv')
                    result_df = pd.DataFrame(columns=['id', 'date', 'position', 'order_price', 'close_price', 'loss_gain'])
                    print(datetime.now())
                    print('予測モデルを更新しました。')
                    line_notify.send('予測モデルを更新しました。')
                else:
                    print('運用成績がベンチマークを上回っているため予測モデルの更新はしません。')
                    line_notify.send('運用成績がベンチマークを上回っているため予測モデルの更新はしません。')
    #else:
    #    minutes = 60 - datetime.now().minute
    #    print('****************************************************************')
    #    print(datetime.now())
    #    print(f'{minutes}分スリープします。\n')
    #    #time.sleep(sleep_time)
    #    for i in range(minutes):
    #        time.sleep(60)
    #        print(datetime.now())
    #        print(f'{i+1}分経過')
    #        print('---------------------------------------------------------')
    #        print('---------------------------------------------------------')
    #        if hour != datetime.now().hour:
    #            continue
    #    print('****************************************************************')
        
line_notify.send('取引ボットの稼働を終了します。')
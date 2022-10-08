import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


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


def data_cleaning(df):
    '''欠損値の補完と目的変数の作成'''
    df.fillna(method='ffill', inplace=True)
    df[df.columns] = df[df.columns].astype(float)
    df['return'] = np.log(df['close'] / df['open'])
    df['sign'] = df['return'].apply(lambda x: 1 if x >= 0 else -1)
    return df

def feature_engineering(df, seqence_width=20):
    '''学習データの特徴量生成'''
    open = df['open'][:-1].values
    close = df['close'][:-1].values
    high = df['high'][:-1].values
    low = df['low'][:-1].values
    eth_open = df['eth_open'][:-1].values
    eth_close = df['eth_close'][:-1].values

    seqence_width = seqence_width
    open_df = sliding_window_view(open, seqence_width)
    close_df = sliding_window_view(close, seqence_width)
    high_df = sliding_window_view(high, seqence_width)
    low_df = sliding_window_view(low, seqence_width)
    eth_open_df = sliding_window_view(eth_open, seqence_width)
    eth_close_df = sliding_window_view(eth_close, seqence_width)

    x_open = open_df[:, :, np.newaxis]
    x_close = close_df[:, :, np.newaxis]
    x_high = high_df[:, :, np.newaxis]
    x_low = low_df[:, :, np.newaxis]
    x_eth_open = eth_open_df[:, :, np.newaxis]
    x_eth_close = eth_close_df[:, :, np.newaxis]

    x_data = np.concatenate([x_open, x_close], axis=2)
    x_data = np.concatenate([x_data, x_high], axis=2)
    x_data = np.concatenate([x_data, x_low], axis=2)
    x_data = np.concatenate([x_data, x_eth_open], axis=2)
    x_data = np.concatenate([x_data, x_eth_close], axis=2)

    x_data = normalise_windows(x_data)
    y_data = df['return'][seqence_width:]
    return x_data, y_data

def feature_engineering_lr(df):
    df['y'] = df['sign'].shift(-1)
    df['open_sma_7'] = df['open'].rolling(7).mean()
    df['open_sma_20'] = df['open'].rolling(20).mean()
    df['close_sma_7'] = df['close'].rolling(7).mean()
    df['close_sma_20'] = df['close'].rolling(20).mean()
    df['eth_open_sma_7'] = df['eth_open'].rolling(7).mean()
    df['eth_open_sma_20'] = df['eth_open'].rolling(20).mean()
    df['eth_close_sma_7'] = df['eth_close'].rolling(7).mean()
    df['eth_close_sma_20'] = df['eth_close'].rolling(20).mean()
    for i in range(1, 6):
        open_col = f'open_{i}lag'
        close_col = f'close_{i}lag'
        eth_open_col = f'eth_open_{i}lag'
        eth_close_col = f'eth_close_{i}lag'
        df[open_col] = df['open'].shift(i)
        df[close_col] = df['close'].shift(i)
        df[eth_open_col] = df['eth_open'].shift(i)
        df[eth_close_col] = df['eth_close'].shift(i)
    columns = ['open', 'high', 'low', 'close', 'eth_open', 'eth_high', 'eth_low', 'eth_close',
           'open_1lag', 'open_2lag', 'open_3lag', 'open_4lag', 'open_5lag', 
           'close_1lag', 'close_2lag', 'close_3lag', 'close_4lag', 'close_5lag',
           'eth_open_1lag', 'eth_open_2lag', 'eth_open_3lag', 'eth_open_4lag', 'eth_open_5lag',
           'eth_close_1lag', 'eth_close_2lag', 'eth_close_3lag', 'eth_close_4lag', 'eth_close_5lag',
           'open_sma_7', 'open_sma_20', 'close_sma_7', 'close_sma_20', 'eth_open_sma_7', 'eth_open_sma_20', 'eth_close_sma_7', 'eth_close_sma_20']
    X = df[columns].iloc[20:-1]
    y = df['y'].iloc[20:-1]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    return x_train, x_test, y_train, y_test

def feature_engineering_svc(df):
    df['y'] = df['sign'].shift(-1)
    df['open_sma_7'] = df['open'].rolling(7).mean()
    df['open_sma_20'] = df['open'].rolling(20).mean()
    df['close_sma_7'] = df['close'].rolling(7).mean()
    df['close_sma_20'] = df['close'].rolling(20).mean()
    df['eth_open_sma_7'] = df['eth_open'].rolling(7).mean()
    df['eth_open_sma_20'] = df['eth_open'].rolling(20).mean()
    df['eth_close_sma_7'] = df['eth_close'].rolling(7).mean()
    df['eth_close_sma_20'] = df['eth_close'].rolling(20).mean()
    for i in range(1, 6):
        open_col = f'open_{i}lag'
        close_col = f'close_{i}lag'
        eth_open_col = f'eth_open_{i}lag'
        eth_close_col = f'eth_close_{i}lag'
        df[open_col] = df['open'].shift(i)
        df[close_col] = df['close'].shift(i)
        df[eth_open_col] = df['eth_open'].shift(i)
        df[eth_close_col] = df['eth_close'].shift(i)
    columns = ['open', 'high', 'low', 'close', 'eth_open', 'eth_high', 'eth_low', 'eth_close',
           'open_1lag', 'open_2lag', 'open_3lag', 'open_4lag', 'open_5lag', 
           'close_1lag', 'close_2lag', 'close_3lag', 'close_4lag', 'close_5lag',
           'eth_open_1lag', 'eth_open_2lag', 'eth_open_3lag', 'eth_open_4lag', 'eth_open_5lag',
           'eth_close_1lag', 'eth_close_2lag', 'eth_close_3lag', 'eth_close_4lag', 'eth_close_5lag',
           'open_sma_7', 'open_sma_20', 'close_sma_7', 'close_sma_20', 'eth_open_sma_7', 'eth_open_sma_20', 'eth_close_sma_7', 'eth_close_sma_20']
    X = df[columns].iloc[20:-1]
    y = df['y'].iloc[20:-1]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    return x_train, x_test, y_train, y_test


def create_x_for_prediction(df, seqence_width=20):
    '''予測値算出のためのデータセット作成'''
    open = df['open'].values
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    eth_open = df['eth_open'].values
    eth_close = df['eth_close'].values

    seqence_width = seqence_width
    open_df = sliding_window_view(open, seqence_width)
    close_df = sliding_window_view(close, seqence_width)
    high_df = sliding_window_view(high, seqence_width)
    low_df = sliding_window_view(low, seqence_width)
    eth_open_df = sliding_window_view(eth_open, seqence_width)
    eth_close_df = sliding_window_view(eth_close, seqence_width)

    x_open = open_df[:, :, np.newaxis]
    x_close = close_df[:, :, np.newaxis]
    x_high = high_df[:, :, np.newaxis]
    x_low = low_df[:, :, np.newaxis]
    x_eth_open = eth_open_df[:, :, np.newaxis]
    x_eth_close = eth_close_df[:, :, np.newaxis]

    x_data = np.concatenate([x_open, x_close], axis=2)
    x_data = np.concatenate([x_data, x_high], axis=2)
    x_data = np.concatenate([x_data, x_low], axis=2)
    x_data = np.concatenate([x_data, x_eth_open], axis=2)
    x_data = np.concatenate([x_data, x_eth_close], axis=2)

    x_data = normalise_windows(x_data)
    return x_data

def create_x_for_lr_prediction(df):
    df['open_sma_7'] = df['open'].rolling(7).mean()
    df['open_sma_20'] = df['open'].rolling(20).mean()
    df['close_sma_7'] = df['close'].rolling(7).mean()
    df['close_sma_20'] = df['close'].rolling(20).mean()
    df['eth_open_sma_7'] = df['eth_open'].rolling(7).mean()
    df['eth_open_sma_20'] = df['eth_open'].rolling(20).mean()
    df['eth_close_sma_7'] = df['eth_close'].rolling(7).mean()
    df['eth_close_sma_20'] = df['eth_close'].rolling(20).mean()
    for i in range(1, 6):
        open_col = f'open_{i}lag'
        close_col = f'close_{i}lag'
        eth_open_col = f'eth_open_{i}lag'
        eth_close_col = f'eth_close_{i}lag'
        df[open_col] = df['open'].shift(i)
        df[close_col] = df['close'].shift(i)
        df[eth_open_col] = df['eth_open'].shift(i)
        df[eth_close_col] = df['eth_close'].shift(i)
    columns = ['open', 'high', 'low', 'close', 'eth_open', 'eth_high', 'eth_low', 'eth_close',
           'open_1lag', 'open_2lag', 'open_3lag', 'open_4lag', 'open_5lag', 
           'close_1lag', 'close_2lag', 'close_3lag', 'close_4lag', 'close_5lag',
           'eth_open_1lag', 'eth_open_2lag', 'eth_open_3lag', 'eth_open_4lag', 'eth_open_5lag',
           'eth_close_1lag', 'eth_close_2lag', 'eth_close_3lag', 'eth_close_4lag', 'eth_close_5lag',
           'open_sma_7', 'open_sma_20', 'close_sma_7', 'close_sma_20', 'eth_open_sma_7', 'eth_open_sma_20', 'eth_close_sma_7', 'eth_close_sma_20']
    X = df[columns].iloc[len(df)-1:, :]
    return X, df[columns]

def create_x_for_svc_prediction(df):
    df['open_sma_7'] = df['open'].rolling(7).mean()
    df['open_sma_20'] = df['open'].rolling(20).mean()
    df['close_sma_7'] = df['close'].rolling(7).mean()
    df['close_sma_20'] = df['close'].rolling(20).mean()
    df['eth_open_sma_7'] = df['eth_open'].rolling(7).mean()
    df['eth_open_sma_20'] = df['eth_open'].rolling(20).mean()
    df['eth_close_sma_7'] = df['eth_close'].rolling(7).mean()
    df['eth_close_sma_20'] = df['eth_close'].rolling(20).mean()
    for i in range(1, 6):
        open_col = f'open_{i}lag'
        close_col = f'close_{i}lag'
        eth_open_col = f'eth_open_{i}lag'
        eth_close_col = f'eth_close_{i}lag'
        df[open_col] = df['open'].shift(i)
        df[close_col] = df['close'].shift(i)
        df[eth_open_col] = df['eth_open'].shift(i)
        df[eth_close_col] = df['eth_close'].shift(i)
    columns = ['open', 'high', 'low', 'close', 'eth_open', 'eth_high', 'eth_low', 'eth_close',
           'open_1lag', 'open_2lag', 'open_3lag', 'open_4lag', 'open_5lag', 
           'close_1lag', 'close_2lag', 'close_3lag', 'close_4lag', 'close_5lag',
           'eth_open_1lag', 'eth_open_2lag', 'eth_open_3lag', 'eth_open_4lag', 'eth_open_5lag',
           'eth_close_1lag', 'eth_close_2lag', 'eth_close_3lag', 'eth_close_4lag', 'eth_close_5lag',
           'open_sma_7', 'open_sma_20', 'close_sma_7', 'close_sma_20', 'eth_open_sma_7', 'eth_open_sma_20', 'eth_close_sma_7', 'eth_close_sma_20']
    X = df[columns].iloc[len(df)-1:, :]
    return X, df[columns]
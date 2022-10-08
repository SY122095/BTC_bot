import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import backend as K

def mish(x):
    '''活性化関数の定義'''
    # get_custom_objects().update({'mish': mish})
    return tf.keras.layers.Lambda(lambda x: x*K.tanh(K.softplus(x)))(x)


def WaveNetResidualConv1D(num_filters, kernel_size, stacked_layer):
    '''WaveNetの構築'''
    def build_residual_block(l_input):
        resid_input = l_input
        for dilation_rate in [2**i for i in range(stacked_layer)]:
            l_sigmoid_conv1d = tf.keras.layers.Conv1D(num_filters, kernel_size, dilation_rate=dilation_rate, padding='same', activation='sigmoid')(l_input)
            l_tanh_conv1d = tf.keras.layers.Conv1D(num_filters, kernel_size, dilation_rate=dilation_rate, padding='same', activation='mish')(l_input)
            l_input = tf.keras.layers.Multiply()([l_sigmoid_conv1d, l_tanh_conv1d])
            l_input = tf.keras.layers.Conv1D(num_filters, 1, padding='same')(l_input)
            resid_input = tf.keras.layers.Add()([resid_input, l_input])
        return resid_input
    return build_residual_block


def data_split(x, y):
    '''学習データ、検証データ、評価データに分割'''
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
    x_tr, x_valid, y_tr, y_valid = train_test_split(x_train, y_train, test_size=0.2, shuffle=False)
    return x_tr, x_valid, x_test, y_tr, y_valid, y_test

def wavenet_training(x_train, x_valid, y_train, y_valid, num_filters=16, kernel_size=8, batchsize=128, lr=0.0001):
    '''wavenetによる学習'''
    num_filters_ = num_filters
    kernel_size_ = kernel_size
    stacked_layers_ = [20, 12, 8, 4, 1]
    shape_ = (None, x_train.shape[2])
    l_input = Input(shape=(shape_))
    x = tf.keras.layers.Conv1D(num_filters_, 1, padding='same')(l_input)
    x = WaveNetResidualConv1D(num_filters_, kernel_size_, stacked_layers_[0])(x)
    x = tf.keras.layers.Conv1D(num_filters_*2, 1, padding='same')(l_input)
    x = WaveNetResidualConv1D(num_filters_*2, kernel_size_, stacked_layers_[1])(x)
    x = tf.keras.layers.Conv1D(num_filters_*4, 1, padding='same')(l_input)
    x = WaveNetResidualConv1D(num_filters_*4, kernel_size_, stacked_layers_[2])(x)
    x = tf.keras.layers.Conv1D(num_filters_*8, 1, padding='same')(l_input)
    x = WaveNetResidualConv1D(num_filters_*8, kernel_size_, stacked_layers_[3])(x)
    x = tf.keras.layers.Conv1D(num_filters_*16, 1, padding='same')(l_input)
    x = WaveNetResidualConv1D(num_filters_*16, kernel_size_, stacked_layers_[4])(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    l_output = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=[l_input], outputs=[l_output])
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='mse', optimizer=optimizer)
    es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(x_train, y_train, validation_data=[x_valid, y_valid], callbacks=[es_callback], epochs=200, batch_size=batchsize)
    return model

def logistic_training(x_train, y_train):
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    return lr

def svc_training(x_train, y_train):
    svc = SVC()
    svc.fit(x_train, y_train)
    return svc

def display_result(model, x_test, y_test, df):
    '''各種評価指標の算出'''
    print('評価データのMSE')
    print(model.evaluate(x_test, y_test))
    print('---------------------------------------------------')
    print('買い(1.0)と売り(-1.0)の回数')
    print(pd.DataFrame(np.sign(model.predict(x_test))).value_counts())
    y_pred = model.predict(x_test)
    predict_df = pd.DataFrame(y_pred, columns=['predict'])
    predict_df['predict'] = predict_df['predict'].apply(lambda x: 1 if x >= 0 else -1)
    return_df = pd.DataFrame(df[['open', 'return', 'sign']][-y_test.shape[0]:])
    return_df.reset_index(drop=True, inplace=True)
    df2 = pd.concat([return_df, predict_df], axis=1)
    df2['strategy'] = df2['return']*df2['predict']
    print('テスト期間の利益: ', df2['strategy'].sum())
    print('ベンチマーク: ', df2['return'].sum())
    print('Buy and Hold: ', (df2['open'][y_test.shape[0]-1] - df2['open'][1]) / df2['open'][1])
    print('正解率', accuracy_score(df2['predict'], df2['sign']))
    print('再現率', recall_score(df2['predict'], df2['sign']))
    print('適合率', precision_score(df2['predict'], df2['sign']))
    print('fスコア', f1_score(df2['predict'], df2['sign']))
    
def display_result_lr(model, x_test, y_test, df):
    y_test.reset_index(drop=True, inplace=True)
    y_pred = model.predict(x_test)
    y_pred = pd.DataFrame(data=y_pred, columns=['y_pred'])
    result_df = pd.concat([y_test, y_pred], axis=1)
    result_df['actual_sign'] = result_df['y'].apply(lambda x: 1 if x > 0 else -1)
    result_df['predicted_sign'] = result_df['y_pred'].apply(lambda x: 1 if x > 0 else -1)
    tmp_df = df[-len(result_df):]
    tmp_df = tmp_df[['open', 'return']]
    tmp_df.reset_index(drop=True, inplace=True)
    result_df = pd.concat([result_df, tmp_df], axis=1)
    result_df['strategy'] = result_df['predicted_sign'] * result_df['return']
    print('正解率: ', accuracy_score(result_df['actual_sign'], result_df['predicted_sign']))
    print('テスト期間の利益: ', result_df['strategy'].sum())
    print('ベンチマーク: ', result_df['return'].sum())
    print('Buy and Hold: ', (result_df['open'][y_pred.shape[0]-1] - result_df['open'][1]) / result_df['open'][1])
    print(result_df['actual_sign'].value_counts())
    print(result_df['predicted_sign'].value_counts())
    plt.figure(figsize=(8, 5))
    plt.plot(result_df['strategy'].dropna().cumsum(), label='logistic')
    plt.plot(result_df['return'].dropna().cumsum(), label='Benchmark')
    plt.legend()
    plt.title('cum return')
    plt.show()
    
def display_result_svc(model, x_test, y_test, df):
    y_test.reset_index(drop=True, inplace=True)
    y_pred = model.predict(x_test)
    y_pred = pd.DataFrame(data=y_pred, columns=['y_pred'])
    result_df = pd.concat([y_test, y_pred], axis=1)
    result_df['actual_sign'] = result_df['y'].apply(lambda x: 1 if x > 0 else -1)
    result_df['predicted_sign'] = result_df['y_pred'].apply(lambda x: 1 if x > 0 else -1)
    tmp_df = df[-len(result_df):]
    tmp_df = tmp_df[['open', 'return']]
    tmp_df.reset_index(drop=True, inplace=True)
    result_df = pd.concat([result_df, tmp_df], axis=1)
    result_df['strategy'] = result_df['predicted_sign'] * result_df['return']
    print('正解率: ', accuracy_score(result_df['actual_sign'], result_df['predicted_sign']))
    print('テスト期間の利益: ', result_df['strategy'].sum())
    print('ベンチマーク: ', result_df['return'].sum())
    print('Buy and Hold: ', (result_df['open'][y_pred.shape[0]-1] - result_df['open'][1]) / result_df['open'][1])
    print(result_df['actual_sign'].value_counts())
    print(result_df['predicted_sign'].value_counts())
    plt.figure(figsize=(8, 5))
    plt.plot(result_df['strategy'].dropna().cumsum(), label='logistic')
    plt.plot(result_df['return'].dropna().cumsum(), label='Benchmark')
    plt.legend()
    plt.title('cum return')
    plt.show()

def plot_result(model, x_test, y_test, df):
    '''グラフ化'''
    y_pred = model.predict(x_test)
    predict_df = pd.DataFrame(y_pred, columns=['predict'])
    predict_df['predict'] = predict_df['predict'].apply(lambda x: 1 if x >= 0 else -1)
    return_df = pd.DataFrame(df[['open', 'return', 'sign']][-y_test.shape[0]:])
    return_df.reset_index(drop=True, inplace=True)
    df2 = pd.concat([return_df, predict_df], axis=1)
    df2['strategy'] = df2['return']*df2['predict']
    plt.figure(figsize=(8, 5))
    plt.plot(df2['strategy'].dropna().cumsum(), label='WaveNet')
    plt.plot(df2['return'].dropna().cumsum(), label='Benchmark')
    plt.legend()
    plt.title('cum return')
    plt.show()
    plt.figure(figsize=(8, 5))
    plt.hist(y_test, bins=30, alpha=1)
    plt.hist(y_pred, bins=30, alpha=0.5)
    plt.show()
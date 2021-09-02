import numpy as np
import pandas as pd
import talib

epsilon = 1e-6


def IdentifyRsi(df, rsiPeriod=14):
    RSI14 = talib.RSI(df.Close.values, timeperiod=rsiPeriod)
    df['RSI'] = RSI14

    return df


def IdentifyRsi50(df, rsiPeriod=50):
    RSI50 = talib.RSI(df.Close.values, timeperiod=rsiPeriod)
    df['RSI50'] = RSI50

    return df


def IdentifyRsi_fliter2(df, rsiPeriod=14):
    RSI = talib.RSI(df.Close.values, timeperiod=rsiPeriod)
    RSI_2 = talib.SMA(RSI, 2)
    df['RSI_2'] = RSI_2

    return df


def IdentifyRsi_fliter7(df, rsiPeriod=14):
    RSI = talib.RSI(df.Close.values, timeperiod=rsiPeriod)
    RSI_7 = talib.SMA(RSI, 7)
    df['RSI_7'] = RSI_7

    return df


def IdentifyIsLowestRsi(df, n=15):
    df['IsLowestRsi'] = [(1 if (j < df['RSI'].iloc[max(0, i - n):i]).sum() == n else 0) \
                         for i, j in enumerate(df['RSI'])]
    return df


def IdentifyIsHighestRsi(df, n=15):
    df['IsHighestRsi'] = [(1 if (j > df['RSI'].iloc[max(0, i - n):i]).sum() == n else 0) \
                          for i, j in enumerate(df['RSI'])]
    return df


def IdentifyMfi(df, mfiPeriod=14):
    high, low, close, volume = df.High.values, df.Low.values, df.Close.values, df.Volume.values

    MFI14 = talib.MFI(high, low, close, volume, timeperiod=mfiPeriod)
    df['MFI'] = MFI14

    return df


def IdentifyEma(df, emaPeriod):
    close = df.Close.values

    EMA = talib.EMA(close, emaPeriod)
    df['EMA' + str(emaPeriod)] = EMA

    return df


def IdentifyMACDHist(df, fastperiod=12, slowperiod=26, signalperiod=9):
    close = df.Close.values
    macd, macdsignal, macdhist = talib.MACD(close, fastperiod=fastperiod, slowperiod=slowperiod,
                                            signalperiod=signalperiod)
    df['MACD'] = macdhist
    return df


def BollingerBands(df, t=20, sigma_up=1, sigma_dn=1, matype=0):
    df['upper_{}'.format(float(sigma_up))], _, df['lower_{}'.format(float(sigma_up))] = talib.BBANDS(
        df.Close,
        timeperiod=t,
        # number of non-biased standard deviations from the mean
        nbdevup=sigma_up,
        nbdevdn=sigma_dn,
        # Moving average type: simple moving average here,0表示sma
        matype=matype)

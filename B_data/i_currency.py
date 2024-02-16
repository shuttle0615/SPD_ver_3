import ccxt
import pandas as pd
import pickle
import talib
import os

from datetime import datetime
from tqdm import tqdm

class Currency:
    
    def __init__(self, symbol, config):
        # initialize data
        self.symbol = symbol
        self.config = config
        
        # view if saved data exist
        if os.path.exists(config.path.db + '/' + symbol + '.pkl'):
            self.load_data()
        else:
            self.update()
        
        
    def download(self, timeframe):
        # create binance object
        binance = ccxt.binance()  
    
        # check earliest time available
        dummyTime = datetime(2000, 1, 1, 10)
        dummyTime = datetime.timestamp(dummyTime)
        dummyTime = int(dummyTime*1000)   
        dummy = binance.fetch_ohlcv(symbol=self.symbol, timeframe=timeframe, since=dummyTime, limit=10)
    
        self.startTime = dummy[0][0]
        
        # process current Time
        endTime = datetime.now()
        endTime = datetime.timestamp(endTime)
        self.endTime = int(endTime*1000) 
        
        # process time frame
        if timeframe == '1h':
            unit_time = (60*60*1000)
        elif timeframe == '30m':
            unit_time = (30*60*1000)
        elif timeframe == '15m':
            unit_time = (15*60*1000)
        elif timeframe == '5m':
            unit_time = (5*60*1000)
        else:
            raise NameError("not supported time frame")
        
        # compute the length btw start and end time (how much data points requried?)
        diff = self.endTime - self.startTime
        num_unit_time = diff // unit_time
        
        # only 1000 data points are available per requrest
        if num_unit_time > 1000 :
            # over 1000 data points

            repeat = num_unit_time // 1000 
            leftover = num_unit_time % 1000 

            ohlcv = []

            for i in tqdm(range(repeat)):
                ohlcv = ohlcv + binance.fetch_ohlcv(symbol=self.symbol, timeframe=timeframe, since=(self.startTime + i*unit_time*1000), limit=1000)

            ohlcv = ohlcv + binance.fetch_ohlcv(symbol=self.symbol, timeframe=timeframe, since=(self.startTime + repeat*unit_time*1000), limit=leftover)    # if leftover is 0, just finish...

        else: 
            # under 1000 data points
            ohlcv = binance.fetch_ohlcv(symbol=self.symbol, timeframe=timeframe, since=self.startTime, limit=num_unit_time)
            
        # process the data
        df = pd.DataFrame(ohlcv, columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
        
        return df
    
    def preprocess(self, df):
        # Require normalization
        
        # Moving average 5, 10, 20, 60, 120
        df['MA_5'] = talib.SMA(df['Close'], timeperiod=5)
        df['MA_10'] = talib.SMA(df['Close'], timeperiod=10)
        df['MA_20'] = talib.SMA(df['Close'], timeperiod=20)
        df['MA_60'] = talib.SMA(df['Close'], timeperiod=60)
        df['MA_120'] = talib.SMA(df['Close'], timeperiod=120)
        
        # MACD 
        macd, macd_signal, macd_hist = talib.MACD(df["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD_main'] = macd
        df['MACD_signal'] = macd_signal
        df['MACD_hist'] = macd_hist
        
        # BBAND
        upper_band, _, Lower_band = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df['Bolinger_up'] = upper_band
        df['Bolinger_down'] = Lower_band
        
        # Momentum
        df['Momentum'] = talib.MOM(df['Close'], timeperiod=10)
        
        # OBV
        df['OBV'] = talib.OBV(df['Close'], df['Volume'])
        
        # CCI
        df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
        
        # AD
        df['AD'] = talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Does not require normalization
        
        # RSI
        df['RSI_6'] = ((talib.RSI(df['Close'], timeperiod=6)) / 100)
        df['RSI_12'] = ((talib.RSI(df['Close'], timeperiod=12)) / 100)
        df['RSI_24'] = ((talib.RSI(df['Close'], timeperiod=24)) / 100)
        
        # MFI
        df['MFI_7'] = ((talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=7)) / 100)
        df['MFI_14'] = ((talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=14)) / 100)
        
        # stochastic
        slowk, slowd = talib.STOCH(df["High"], df["Low"], df["Close"], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        df['STOCH_k'] = ((slowk) / 100)
        df['STOCH_d'] = ((slowd) / 100)
        
        # Ultimate Oscillator
        df['ULTOSC'] = ((talib.ULTOSC(df['High'], df['Low'], df['Close'])) / 100)
        
        # WR
        df['WILLR'] = ((talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)) / 100)
        
        # AROON
        df['AROON'] = ((talib.AROONOSC(df['High'], df['Low'], timeperiod=14)) / 100)
        
        
        # return over forward_window period
        for i in self.forward:
            df[f'r_value_{i}'] = (df['Close'].shift(-1 * i) - df['Open'].shift(-1)) / (df['Open'].shift(-1))
        
        return df
    
    def save_data(self):
        with open(self.config.path.db + '/' + self.symbol + '.pkl', 'wb') as file:
            pickle.dump(self, file)
    
    def load_data(self):
        with open(self.config.path.db + '/' + self.symbol + '.pkl', 'rb') as file:
            loaded_obj = pickle.load(file)
            self.__dict__.update(loaded_obj.__dict__)
        
    def update(self):
        # make states easy to read
        self.timeframe = self.config.data.timeframe
        self.forward = self.config.data.forward_window
        
        # for each timeframe
        for tf in self.timeframe:
            print(f"start downloading {self.symbol} with timeframe {tf}")
            data = self.download(tf)
            data = self.preprocess(data)
            setattr(self, f'data_{tf}', data)
        
        # save processed data
        self.save_data()
        
    def remove(self):
        if os.path.exists(self.config.path.db + '/' + self.symbol + '.pkl'):
            print(f'removing {self.symbol}')
            os.remove(self.config.path.db + '/' + self.symbol + '.pkl')

    def retrive(self, timeframe, start, end):
        # start in unix time stamp
        startTime = datetime(*start)
        startTime = datetime.timestamp(startTime)
        startTime = int(startTime*1000)
        
        # end in unix time stamp
        endTime = datetime(*end)
        endTime = datetime.timestamp(endTime)
        endTime = int(endTime*1000)
        
        if timeframe == '1h':
            df = self.data_1h[
                (self.data_1h['Time'] > startTime) &
                (self.data_1h['Time'] < endTime)].reset_index(drop=True)
        elif timeframe == '30m':
            df = self.data_30m[
                (self.data_30m['Time'] > startTime) &
                (self.data_30m['Time'] < endTime)].reset_index(drop=True)
        elif timeframe == '15m':
            df = self.data_15m[
                (self.data_15m['Time'] > startTime) &
                (self.data_15m['Time'] < endTime)].reset_index(drop=True)
        elif timeframe == '5m':
            df = self.data_5m[
                (self.data_5m['Time'] > startTime) &
                (self.data_5m['Time'] < endTime)].reset_index(drop=True)
        else:
            raise NameError("not supported time frame")
        return df

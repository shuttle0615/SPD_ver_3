import ccxt
import numpy as np
from datetime import datetime
from tqdm import tqdm 
from omegaconf import OmegaConf
import time
import os

from .i_currency import Currency

def current_coin_list():
    # list of coins to be tracked

    binance = ccxt.binance() 
    ls = binance.fetchMarkets()

    # show all active markets
    active = [market for market in ls if market['active']]

    # show all markets with usdt
    asset = [market['id'] for market in active if market['id'].endswith("USDT")]
    asset = np.unique(np.array(asset))

    # get volume and percentage change of today
    print('getting stats of the coins')
    asset_stats = []
    for coin in tqdm(asset):
        data = binance.fetch_ticker(coin)
        p_chage = data['percentage']
        volume = data['quoteVolume']
        time.sleep(0.1)
        
        asset_stats.append((volume, p_chage, coin))
        
    # filtering by percentage change to get rid of Stable coins
    processed_asset_stats = []
    for stats in asset_stats:
        if (abs(stats[1]) > 0.01):
            processed_asset_stats.append(stats)

    # sort it by volume
    processed_asset_stats = sorted(processed_asset_stats, reverse=True)

    # only records the coins existed before 2020
    print('check the trading start day')
    at_least_since_2019 = []
    for coin in tqdm(processed_asset_stats):
        # fetch earliest time
        dummyTime = datetime(2000, 1, 1, 10)
        dummyTime = datetime.timestamp(dummyTime)
        dummyTime = int(dummyTime*1000)   
        dummy = binance.fetch_ohlcv(symbol=coin[2], timeframe='5m', since=dummyTime, limit=10)

        startTime = dummy[0][0]
        
        # filter based on earliest time
        if startTime < int(datetime.timestamp(datetime(2020, 1, 1, 10))*1000):
            at_least_since_2019.append(coin)
    
    return [coin[2] for coin in at_least_since_2019]

def db_coin_list(config):
    # get all name lists from db path   
    entries = os.listdir(config.path.db)
    return [coin.replace('.pkl', '') for coin in entries]

def create_db():
    # load data conifg
    config = OmegaConf.load('/content/drive/MyDrive/Auto_trading/SPD_ver_3/A_config/data_config.yaml')
    
    # get current coin list
    current_coin = current_coin_list()
    
    # download all coins
    for coin in current_coin:
        Currency(coin, config) 

def update_db():
    # load data conifg
    config = OmegaConf.load('/content/drive/MyDrive/Auto_trading/SPD_ver_3/A_config/data_config.yaml')
    
    # get current coin list
    current_coin = current_coin_list()
    
    # get current db coin list 
    db_coin = db_coin_list(config)
    
    # remove out dated coins 
    outdated = list(set(db_coin) - set(current_coin))
    print(f'removing {len(outdated)} Currencies')
    for coin in outdated:
        Currency(coin, config).remove()
    
    # update old coins
    updates = list(set(db_coin) & set(current_coin))
    print(f'updating {len(updates)} Currencies')
    for coin in updates:
        Currency(coin, config).update() 
        
    # download new coins
    new = list(set(current_coin) - set(db_coin))
    print(f'downloading {len(new)} Currencies')
    for coin in new:
        Currency(coin, config) 

def sort_coin_by_size():
    config = OmegaConf.load('/content/drive/MyDrive/Auto_trading/SPD_ver_3/A_config/data_config.yaml')
    
    size_ls = [(file.stat().st_size, file.name) for file in os.scandir(config.path.db)]
    size_ls = sorted(size_ls, reverse=True)
    return [coin[1].replace('.pkl', '') for coin in size_ls]

if __name__ == "__main__":
    #create_db()
    #update_db()
    ls = current_coin_list()
    print(ls)
    print(len(ls))
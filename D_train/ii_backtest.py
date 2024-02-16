from B_data.iii_datamodule import BackTestDataModule
from C_model import *

import lightning.pytorch as pl
from omegaconf import OmegaConf
import pandas as pd 
import json
import os

BUY = 0
HOLD = 1
SELL = 2

def calc_cum_ret_s1(x, stop_loss, fee):
    """
    compute cumulative return strategy s1.
    compute the multiplication factor of original capital after n iteration,
    reinvesting the gains.
    It search for BUY order, execute them and close them when stop_loss hits happen or when reversal/SELL signal.
    Finds the pairs BUY/SELL orders and compute the cumulative return.
    If no BUY order is encountered, it does nothing.
    If no SELL order is encountered, once a BUY order was issued, a dummy SELL order of the position is issued at the end of the period.
    :param x: ordered prices timeseries of a coin and labels
    :param stop_loss:
    :param fee: commission fee applied
    :return: history, capital, num_op, min_drowdown, max_gain
    """

    order_pending = 0
    price_start = 0
    price_end = 0
    capital = 1
    history = []
    labs = x['label'].values  # debug

    min_drowdown = 0
    max_gain = 0
    num_ops = 0
    good_ops = 0
    for row in x.itertuples():

        # handle stop loss
        if order_pending:
            price_end = row.Low
            pct_chg = (price_end - price_start) / price_start
            if pct_chg < -stop_loss:
                order_pending = 0

                price_end = price_start * (1 - stop_loss)

                pct_chg = (price_end - price_start) / price_start
                if pct_chg < min_drowdown:
                    min_drowdown = pct_chg
                
                capital *= 1 + (((price_end * (1 - fee)) -
                                 (price_start * (1 + fee))) /
                                (price_start * (1 + fee)))
                price_start = price_end = 0

        history.append(capital)

        if row.label == BUY:
            if order_pending:
                continue

            else:
                order_pending = 1
                price_start = row.Close
                num_ops += 1
                continue

        if row.label == HOLD:
            continue

        if row.label == SELL:
            if order_pending:
                price_end = row.Close
                pct_chg = (price_end - price_start) / price_start
                if pct_chg > 0:
                    good_ops += 1

                if pct_chg < min_drowdown:
                    min_drowdown = pct_chg

                if pct_chg > max_gain:
                    max_gain = pct_chg

                order_pending = 0
                capital *= 1 + (((price_end * (1 - fee)) -
                                 (price_start * (1 + fee))) /
                                (price_start * (1 + fee)))
                price_start = price_end = 0
                continue

            else:
                continue

    # handle last candle
    if order_pending:
        price_end = row.Low
        pct_chg = (price_end - price_start) / price_start
        if pct_chg < -stop_loss:
            price_end = price_start * (1 - stop_loss)

        if pct_chg < min_drowdown:
            min_drowdown = pct_chg

        if pct_chg > max_gain:
            max_gain = pct_chg

        capital *= 1 + (((price_end * (1 - fee)) -
                         (price_start * (1 + fee))) /
                        (price_start * (1 + fee)))
    

    return history, capital, num_ops, min_drowdown, max_gain, good_ops


def back_test(config, data_config, coin, start, end): 
    # load data
    datamodule = BackTestDataModule(config, data_config, coin, start, end)
    
    # define module
    if config.model == 'transformer':
        model = TransformerModule.load_from_checkpoint(f'{data_config.path.result_path}/{config.name}/setting{config.num}.ckpt')
    elif config.model == 'LSTM':
        model = LSTMModule.load_from_checkpoint(f'{data_config.path.result_path}/{config.name}/setting{config.num}.ckpt')    
    else:
        print('model is not implemented')
        return
    
    # conduct bactest
    trainer = pl.Trainer(logger=None, use_distributed_sampler=False, accelerator="gpu", devices=1)
    predictions = trainer.predict(model=model, datamodule=datamodule)
    
    # (raw logit float 64 size(64, 3), target size(64), idx of dataloader)
    
    # extract data
    ind = []
    output = []
    for p in predictions:
        ind = ind + list(p[0].cpu().numpy()) 
        output = output + list(p[1].cpu().numpy())
    
    df = datamodule.df.iloc[ind[0]:ind[-1]+1].copy().reset_index(drop=True)
    df['label'] = pd.Series(output)
    
    # calculate return
    hist_nn, cap_nn, num_op_nn, min_drawdown_nn, max_gain_nn, g_ops_nn = calc_cum_ret_s1(df, config['stop_loss'], config['commission_fee'])
    
    result = {'asset_name': coin, 
            'start': start,
            'end': end,
            'profit_nn': cap_nn,
            'operations': num_op_nn, 
            'max_dropdown': min_drawdown_nn, 
            'max_gain': max_gain_nn, 
            "successfull_operation": g_ops_nn, 
            'history': hist_nn}
    
    file_path = f'{data_config.path.result_path}/{config.name}' + f'/setting{config.num}.json'
    
    # check either file exist or not
    if os.path.exists(file_path):
        # load json file to store result
        with open(file_path, 'r') as file:
            data = json.load(file)
    
        # check the list to see if experiment over that coin exist
    
        for i, r in enumerate(data):
            if r['asset_name'] == coin:
                data[i] = result 
            
                with open(file_path, 'w') as file:
                    json.dump(data, file, indent=4)
            
                return
        
        data.append(result)
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
    
    else:
        # add result and return
        data = [result]
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
            
            
def conduct_backtest(name, coin_ls, exp_ls, start, end):
    
    data_config = OmegaConf.load('/home/kyuholee/SPD_ver_3/A_config/data_config.yaml')
    
    for n in exp_ls:
        for coin in coin_ls:
            print(f'bactesting {coin}: {name} - setting {n}')
            
            config = OmegaConf.load(f'{data_config.path.config_path}/{name}/setting{n}.yaml')

            back_test(config, data_config, coin, start, end)
        
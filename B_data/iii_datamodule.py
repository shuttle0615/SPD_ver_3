from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from torch.utils.data import Dataset, DataLoader
import lightning as pl 
import numpy as np
import pandas as pd

from .i_currency import Currency
from .ii_db_management import sort_coin_by_size
from .iv_imbalance import ImbalancedDatasetSampler

BUY = 0
HOLD = 1
SELL = 2

class stockDataModule(pl.LightningDataModule):
    
    def __init__(self, config, data_config):
        super().__init__()
        
        print('Datamodule Initialization')
        
        # load configurations 
        self.config = config
        self.data_config = data_config
        
        # load coin list
        self.coin_ls = sort_coin_by_size()[:self.config.coin_number]
        
        # for train
        self.df_train, self.df_train_idx = self.idx_generator(self.coin_ls, self.config.train)
        self.df_train_idx = self.sample_and_concatanate(self.df_train_idx, self.config.train)
        self.train_r_value = self.get_r_value(self.df_train, self.df_train_idx) 
        print('train data loaded')
        
        # for validation
        self.df_val, self.df_val_idx = self.idx_generator(self.coin_ls, self.config.validation)
        self.df_val_idx = self.sample_and_concatanate(self.df_val_idx, self.config.validation)
        self.val_r_value = self.get_r_value(self.df_val, self.df_val_idx)
        print('validation data loaded')
        
        # for test
        self.df_test, self.df_test_idx = self.idx_generator(self.coin_ls, self.config.test)
        self.df_test_idx = self.sample_and_concatanate(self.df_test_idx, self.config.test)
        self.test_r_value = self.get_r_value(self.df_test, self.df_test_idx)
        print('test data loaded')
        
        # compute alphas (use train and validation only)
        self.alpha = pd.Series(self.train_r_value + self.val_r_value).abs().quantile(self.config.quantile)
        
    def setup(self, stage=None):
        # load train dataset
        self.train = StockDataset(self.df_train, self.df_train_idx, self.train_r_value, self.alpha)
        # load validation dataset
        self.val = StockDataset(self.df_val, self.df_val_idx, self.val_r_value, self.alpha)
        # load test dataset
        self.test = StockDataset(self.df_test, self.df_test_idx, self.test_r_value, self.alpha)
        
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.config.batch, num_workers=6, sampler=ImbalancedDatasetSampler(self.train, labels=self.train.label))
    
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.config.batch, num_workers=6, sampler=ImbalancedDatasetSampler(self.val, labels=self.val.label))
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.config.batch, num_workers=6, sampler=ImbalancedDatasetSampler(self.test, labels=self.test.label))
    
    def idx_generator(self, coin_ls, config):
        df_ls = []
        df_ls_idx = []
        
        for i, coin in enumerate(coin_ls):
            # train
            
            # load data
            df = Currency(coin, self.data_config).retrive(self.config.timestamp, tuple(config.start), tuple(config.end))
            df_ls.append(df)
            
            # generate idx pair 
            df_ls_idx.append([(idx - self.config.backward, idx, i) for idx in range(self.config.backward, len(df), self.config.sliding)])
            
        return df_ls, df_ls_idx
    
    def sample_and_concatanate(self, df_ls_idx, config):
        # how many samples per coin
        n = config.size // len(df_ls_idx)
        
        temp = []
        for ls_idx in df_ls_idx:
            indices = np.random.choice(len(ls_idx), n, replace=False)
            sampled_tuples = [ls_idx[i] for i in indices]
            temp = temp + sampled_tuples
        
        return temp
    
    def get_r_value(self, df_ls, df_ls_idx):
        return [df_ls[idx_tuple[2]][f'r_value_{self.config.forward}'][idx_tuple[1]] for idx_tuple in df_ls_idx]
        
class StockDataset(Dataset):        
    def __init__(self, df, idx, r_value, alpha):
        super().__init__()
        self.data = df
        self.idx = idx
        self.r_value = pd.Series(r_value)
        self.label = pd.Series(1, index=self.r_value.index)
        self.label[self.r_value > alpha] = 0
        self.label[self.r_value < (-1 * alpha)] = 2
        
    def __len__(self):
        return len(self.idx)
    
    def __getitem__(self, index):
        start, end, coin = self.idx[index]
        X = index_to_value(self.data[coin], start, end)
        y = self.label[index]
        return X, y
        
class BackTestDataModule(pl.LightningDataModule):
    def __init__(self, config, data_config, coin, start, end):
        super().__init__()
        
        self.config = config
        self.data_config = data_config
        
        self.coin = coin
        
        self.df = Currency(self.coin, self.data_config).retrive(self.config.timestamp, tuple(start), tuple(end))
        
    def setup(self, stage=None):
        self.back_test = BackTestDataset(self.df, self.config.backward, self.config.sliding, self.coin)
        
    def predict_dataloader(self):
        return DataLoader(self.back_test, batch_size=128, num_workers=6) 

        
class BackTestDataset(Dataset):
    def __init__(self, df, backward, sliding, name):
        super().__init__()
        self.name = name
        self.data = df
        self.idx = [(idx - backward, idx) for idx in range(backward, len(df), sliding)]
        
    def __len__(self):
        return len(self.idx)
    
    def __getitem__(self, index):
        start, end = self.idx[index]
        X = index_to_value(self.data, start, end)
        return X, self.idx[index][1]
    
    
def index_to_value(data, start, end):

    def normalize(data):
        mue = np.mean(data)
        std = np.std(data)

        if std == 0.0:
            data = data - mue
        else:
            data = (data - mue) / std
        return data

    def log_normalize(data):
        mue = np.mean(data)
        std = np.std(data)

        if std == 0.0:
            data = data - mue
        else:
            data = (data - mue) / std
        return data
        

    # data with normalization
    actual_data = data.iloc[start:end, [1,2,3,4,5,6,7,8,9,10,14,15]].values
    actual_data = normalize(actual_data)

    MACD_data = data.iloc[start:end, [11,12,13]].values
    MACD_data = normalize(MACD_data)

    MOM_data = data.iloc[start:end, [16]].values
    MOM_data = normalize(MOM_data)

    OBV_data = data.iloc[start:end, [17]].values
    OBV_data = normalize(OBV_data)

    CCI_data = data.iloc[start:end, [18]].values
    CCI_data = normalize(CCI_data)

    AD_data = data.iloc[start:end, [19]].values
    AD_data = normalize(AD_data)

    # data without normalization
    data = data.iloc[start:end, 20:-4].values

    # can mainpulate which features to use.
    return np.hstack((actual_data, MACD_data, MOM_data, OBV_data, CCI_data, AD_data, data)).astype(np.float32)
import yaml
import os
from omegaconf import OmegaConf
    
config = OmegaConf.load('/home/kyuholee/SPD_ver_3/A_config/data_config.yaml')

def generate_yaml(dic, path):
    num = len(os.listdir(path + '/' + dic['name']))
    dic['num'] = num

    with open((path + f'/{dic["name"]}/setting{num}.yaml'), 'w') as outfile:
        yaml.dump(dic, outfile, default_flow_style=False)            

for lr in [1.0e-5, 1.0e-5 / 2, 1.0e-5 / 4, 1.0e-5 / 8, 1.0e-5 / 16, 1.0e-5 / 32, 1.0e-5 / 64, 1.0e-5 / 128]: #, 100000, 75000, 25000
    dic = {
        # data section
        'name' : 'exp_7',
        'coin_number' : 40,
        'timestamp' : '5m',
        'sliding' : 1,
        'forward': 5, # 5 is the max
        'backward': 100,
        'quantile' : 0.85,
        'batch': 128,
        
        'train': {
            'start' : [2020, 5, 1, 10],
            'end' : [2022, 7, 1, 10],
            'size' : 100000
        },
        
        'validation': {
            'start' : [2022, 9, 1, 10],
            'end' : [2023, 2, 1, 10],
            'size' : 5000
        },  
        
        'test': {
            'start' : [2023, 6, 1, 10],
            'end' : [2023, 9, 1, 10],
            'size' : 5000
        },
        
        # model section
        'model': 'LSTM',
        
        'transformer': {
            "input_dim": 29,
            "nhid_tran" : 512, #model
            "nhead" : 32, #model
            "nlayers_transformer" : 8, #model
            "attn_pdrop" : 0.1, #model
            "resid_pdrop" : 0.1, #model
            "embd_pdrop" : 0.1, #model
            "nff" : 4 * 512, #model
            "n_class": 3
        },
        
        'CNN_transformer': {},
        'LSTM': {
            'ninp': 29,
            'nhid': 256,
            'nlayers': 8,
            'dropout': 0.1,
            'n_class':3
        },
        'CNN_LSTM': {},
        
        # train section
        'epoch': 10,
        'lr': lr,
        'scheduler': None,
            
            #{
            #    'name': 'MultiStepLR',
            #    'milestones': [5, 12, 20], 
            #    'gamma': 0.5
            #}, 
            #{
            #   name: 'ReduceLROnPlateau' 
            #   mode: 'min', 
            #   factor: 0.5, 
            #   patience: 1, 
            #   threshold: 0.001, 
            #   threshold_mode: 'rel', 
            #   cooldown: 0, 
            #   min_lr: 0.00001,
            #   verbose: True   
            #}
        
        # backtest section
        'stop_loss': 0.01,
        'commission_fee': 0.001
    }

    generate_yaml(dic, config.path.config_path)

'''
NOTE
1. since validation and test are about same range, and has both bullish and bearish movement, they must show similar behavior. 
2. coins can be 

'''
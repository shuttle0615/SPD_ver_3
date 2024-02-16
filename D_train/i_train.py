import wandb
import os
from omegaconf import OmegaConf

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from B_data.iii_datamodule import stockDataModule
from C_model import *

def train(config, data_config):
    # load dataset
    datamodule = stockDataModule(config, data_config)
    
    # define module
    if config.model == 'transformer':
        model = TransformerModule(config)
    elif config.model == 'LSTM':
        model = LSTMModule(config)
    else:
        print('model is not implemented')
        return
    
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="validation_loss",
        mode="min",
        dirpath=(f'{data_config.path.result_path}/{config.name}'),
        filename= f'setting{config.num}'
    )
    
    wandb_logger = WandbLogger(name=f'{config.name}_setting{config.num}', project="SPD_ver_3", save_dir=f'{data_config.path.result_path}/{config.name}')
    
    trainer = Trainer(
        accelerator="gpu", 
        max_epochs=config.epoch,
        log_every_n_steps=10,
        logger=wandb_logger, 
        default_root_dir=f'{data_config.path.result_path}',
        callbacks=[checkpoint_callback],
        use_distributed_sampler=False)

    trainer.fit(model=model, datamodule=datamodule)
    
    trainer.test(datamodule=datamodule)
    
    wandb.finish()

def conduct_experiment(name, li):
    # conduct experiment by name
    
    # load data config
    data_config = OmegaConf.load('/home/kyuholee/SPD_ver_3/A_config/data_config.yaml')
    
    for n in li:
        print(f'experiment: {name}, setting: setting{n}')
        
        config = OmegaConf.load(f'{data_config.path.config_path}/{name}/setting{n}.yaml')
    
        train(config, data_config)
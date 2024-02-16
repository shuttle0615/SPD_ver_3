import numpy as np

from typing import Any
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import torchmetrics

MAX_LEN = 100
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class MaskedMultiheadAttention(nn.Module):
    """
    A vanilla multi-head masked attention layer with a projection at the end.
    """
    def __init__(self, config, mask=False):
        super(MaskedMultiheadAttention, self).__init__()
        assert config["nhid_tran"] % config["nhead"] == 0
        # mask : whether to use
        # key, query, value projections for all heads
        self.key = nn.Linear(config["nhid_tran"], config["nhid_tran"])
        self.query = nn.Linear(config["nhid_tran"], config["nhid_tran"])
        self.value = nn.Linear(config["nhid_tran"], config["nhid_tran"])
        # regularization
        self.attn_drop = nn.Dropout(config["attn_pdrop"])
        # output projection
        self.proj = nn.Linear(config["nhid_tran"], config["nhid_tran"])
        # causal mask to ensure that attention is only applied to the left in the input sequence
        if mask:
            self.register_buffer("mask", torch.tril(torch.ones(MAX_LEN, MAX_LEN)))
        self.nhead = config["nhead"]
        self.d_k = config["nhid_tran"] // config["nhead"]

    def forward(self, q, k, v, mask=None):
        # WRITE YOUR CODE HERE

        Q = self.query(q)
        Q_size = Q.size()
        Q = Q.reshape([Q_size[0], Q_size[1], self.nhead, -1])
        Q = torch.transpose(Q, 1, 2)

        K = self.key(k)
        K_size = K.size()
        K = K.reshape([K_size[0], K_size[1], self.nhead, -1])
        K = torch.transpose(K, 1, 2)

        V = self.value(v)
        V_size = V.size()
        V = V.reshape([V_size[0], V_size[1], self.nhead, -1])
        V = torch.transpose(V, 1, 2)

        K = torch.transpose(K, 2, 3)
        R = torch.matmul(Q, K)
        R = R/(self.d_k ** (1/2))

        #casual masking
        if hasattr(self, "mask"):
          temp_mask = self.mask[:R.size(2),:R.size(3)]
          temp_mask = temp_mask < 0.5
          R = R.masked_fill_(temp_mask, -float('Inf'))

        #Pad masking
        if mask != None:
          mask = mask < 0.5
          R = R.permute([2,1,0,3])
          R = R.masked_fill_(mask.to(device), -float('Inf'))
          R = R.permute([2,1,0,3])

        R = torch.nn.Softmax(dim=-1)(R)
        R = self.attn_drop(R)
        output = torch.matmul(R, V)

        output = output.transpose(1,2)
        output = output.reshape(output.size(0), output.size(1), -1)
        output = self.proj(output)

        assert output != None , output
        return output


class TransformerEncLayer(nn.Module):
    def __init__(self, config):
        super(TransformerEncLayer, self).__init__()
        self.ln1 = nn.LayerNorm(config["nhid_tran"])
        self.ln2 = nn.LayerNorm(config["nhid_tran"])
        self.attn = MaskedMultiheadAttention(config)
        self.dropout1 = nn.Dropout(config["resid_pdrop"])
        self.dropout2 = nn.Dropout(config["resid_pdrop"])
        self.ff = nn.Sequential(
            nn.Linear(config["nhid_tran"], config["nff"]),
            nn.ReLU(),
            nn.Linear(config["nff"], config["nhid_tran"])
        )

    def forward(self, x, mask=None):
        # WRITE YOUR CODE HERE
        output = self.ln1(x)
        output = output + self.dropout1(self.attn(output, output, output, mask))
        output = self.ln2(output)
        output = output + self.dropout2(self.ff(output))

        return output
    
class PositionalEncoding(nn.Module):
    def __init__(self, config, max_len=4096):
        super().__init__()
        dim = config["nhid_tran"]
        pos = np.arange(0, max_len)[:, None]
        i = np.arange(0, dim // 2)
        denom = 10000 ** (2 * i / dim)

        pe = np.zeros([max_len, dim])
        pe[:, 0::2] = np.sin(pos / denom)
        pe[:, 1::2] = np.cos(pos / denom)
        pe = torch.from_numpy(pe).float()

        self.register_buffer('pe', pe)

    def forward(self, x):
        # DO NOT MODIFY
        # 1 -> 0 but why?
        return x + self.pe[:x.shape[1]]

class TransformerEncoder(nn.Module):

    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        # input embedding stem
        self.tok_emb = nn.Linear(config["input_dim"], config["nhid_tran"]) #ohlev encoding
        self.pos_enc = PositionalEncoding(config)
        self.dropout = nn.Dropout(config["embd_pdrop"])
        # transformer
        self.nlayers_transformer = config["nlayers_transformer"]
        self.transform = nn.ModuleList([TransformerEncLayer(config) for _ in range(config["nlayers_transformer"])])
        # decoder head
        self.ln_f = nn.LayerNorm(config["nhid_tran"])
        self.classifier_head = nn.Sequential(
           nn.Linear(config["nhid_tran"], config["nhid_tran"]),
           nn.LeakyReLU(),
           nn.Dropout(config["embd_pdrop"]),
           nn.Linear(config["nhid_tran"], config["nhid_tran"]),
           nn.LeakyReLU(),
           nn.Linear(config["nhid_tran"], config["n_class"])
       )


    def forward(self, x, mask=None):
        # WRITE YOUR CODE HERE
        output = self.tok_emb(x)
        output = self.pos_enc(output)
        output = self.dropout(output)

        for i in range(self.nlayers_transformer):
          output = self.transform[i](output, mask=mask)

        output = self.ln_f(output)
        output = output.mean(dim=1)
        output = self.classifier_head(output)
        return output


class TransformerModule(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lr = self.config["lr"]
        self.transformer = TransformerEncoder(self.config['transformer'])
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.config['transformer']['n_class'], average=None)
        self.F1 = torchmetrics.F1Score(task="multiclass", num_classes=self.config['transformer']['n_class'])
        self.save_hyperparameters()
        
    def forward(self, inputs):
        return self.transformer(inputs)
    
    def training_step(self, batch, batch_idx):
        # can i get y values here? if so, it can compute ROI 
        inputs, target = batch
        output = self.transformer(inputs)
        loss = torch.nn.functional.cross_entropy(output, target)
        
        prob = torch.nn.functional.softmax(output, dim=1)
        accuracy = self.accuracy(prob, target)
        F1 = self.F1(prob, target)
        
        self.log_dict({"loss":loss, "accuracy_0":accuracy[0], "accuracy_1":accuracy[1], f"accuracy_2":accuracy[2], "F1":F1})
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.transformer(inputs)
        val_loss = torch.nn.functional.cross_entropy(output, target)
        
        prob = torch.nn.functional.softmax(output, dim=1)
        val_accuracy = self.accuracy(prob, target)
        val_F1 = self.F1(prob, target)
        
        self.log_dict({"validation_loss":val_loss, "validation_accuracy_0":val_accuracy[0], "validation_accuracy_1":val_accuracy[1], "validation_accuracy_2":val_accuracy[2], "val_F1":val_F1}, sync_dist=True)
        return val_loss, val_accuracy
    
    def test_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.transformer(inputs)
        test_loss = torch.nn.functional.cross_entropy(output, target)
        
        prob = torch.nn.functional.softmax(output, dim=1)
        test_accuracy = self.accuracy(prob, target)
        test_F1 = self.F1(prob, target)
        
        self.log_dict({"test_loss":test_loss, "test_accuracy_0":test_accuracy[0], "test_accuracy_1":test_accuracy[1], "test_accuracy_2":test_accuracy[2],  "test_F1":test_F1}, sync_dist=True)
        
        return output
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        
        inputs, ind = batch
        
        #outputs = self.transformer(inputs)
        
        #outputs = outputs.view(3)
        
        #outputs = 0 if outputs[0] > 0.85 else 2 if outputs[2] > 0.85 else 1
        
        outputs = torch.argmax(self.transformer(inputs), dim=1) 
        return ind, outputs
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.transformer.parameters(), lr=self.lr)
        
        if self.config['scheduler'] == None:
            return optimizer
        elif self.config['scheduler']['name'] == 'MultiStepLR':
            setting = self.config['scheduler']
            scheduler = MultiStepLR(optimizer,
                milestones=setting['milestones'], 
                gamma=setting['gamma']
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch'
                }
            }
            
        elif self.config['scheduler']['name'] == 'ReduceLROnPlateau':
            setting = self.config['scheduler']
            scheduler = ReduceLROnPlateau(optimizer, 
                mode=setting['mode'], 
                factor=setting['factor'], 
                patience=setting['patience'], 
                threshold=setting['threshold'], 
                threshold_mode=setting['threshold_mode'], 
                cooldown=setting['cooldown'], 
                min_lr=setting['min_lr'],
                verbose=setting['verbose'])
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                    'monitor': 'validation_loss',
                    'strict': True,
                }
            }      
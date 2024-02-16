
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import torchmetrics
from lightning import LightningModule


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_input = nn.Linear(input_size, 4 * hidden_size)
        self.linear_hidden = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, x, state):
        # WRITE YOUR CODE HERE
        hx = state[0]
        cx = state[1]

        hidden = self.linear_hidden(hx)
        input = self.linear_input(x)

        forget, in_val, cell, out = torch.chunk(hidden + input, 4, dim=1)

        cx = cx * torch.sigmoid(forget)
        cx = cx + (torch.sigmoid(in_val) * torch.tanh(cell))

        cy = cx
        hy = torch.tanh(cx) * torch.sigmoid(out)

        return hy, (hy, cy)
    

class LSTMLayer(nn.Module):
    def __init__(self,*cell_args):
        super(LSTMLayer, self).__init__()
        self.cell = LSTMCell(*cell_args)

    def forward(self, x, state, length_x=None):
        # DO NOT MODIFY
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        inputs = x.unbind(0)
        assert (length_x is None) or torch.all(length_x == length_x.sort(descending=True)[0])
        outputs = []
        out_hidden_state = []
        out_cell_state = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i] , state)
            outputs += [out]
            if length_x is not None:
                if torch.any(i+1 == length_x):
                    out_hidden_state = [state[0][i+1==length_x]] + out_hidden_state
                    out_cell_state = [state[1][i+1==length_x]] + out_cell_state
        if length_x is not None:
            state = (torch.cat(out_hidden_state, dim=0), torch.cat(out_cell_state, dim=0))
        return torch.stack(outputs), state


class LSTM(nn.Module):
    def __init__(self, ninp, nhid, num_layers, dropout):
        super(LSTM, self).__init__()
        self.layers = []
        self.dropout = nn.Dropout(dropout)
        for i in range(num_layers):
            if i == 0:
                self.layers.append(LSTMLayer(ninp, nhid))
            else:
                self.layers.append(LSTMLayer(nhid, nhid))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, states, length_x=None):
        # WRITE YOUR CODE HERE
        num_layers = len(self.layers)
        output_states = []
        input = x

        for i in range(num_layers):
          output, out_state = self.layers[i](input, states[i], length_x)
          output_states.append(out_state)
          input = self.dropout(output)

        return input, output_states
    
class LSTMEncoder(nn.Module):
    def __init__(self, args):
        super(LSTMEncoder, self).__init__()
        self.args = args
        self.ninp = args['ninp']
        self.nhid = args['nhid']
        self.nlayers = args['nlayers']
        self.dropout = args['dropout']
        self.lstm = LSTM(self.ninp, self.nhid, self.nlayers, self.dropout)
        self.ln_f = nn.LayerNorm(args["nhid"])
        self.classifier_head = nn.Sequential(
           nn.Linear(args["nhid"], args["nhid"]),
           nn.LeakyReLU(),
           nn.Dropout(args["dropout"]),
           nn.Linear(args["nhid"], args["nhid"]),
           nn.LeakyReLU(),
           nn.Linear(args["nhid"], args["n_class"])
       )
        
    def _get_init_states(self, x):
        init_states = [
            (torch.zeros((x.size(1), self.nhid)).to(x.device),
            torch.zeros((x.size(1), self.nhid)).to(x.device))
            for _ in range(self.nlayers)
        ]
        return init_states

    def forward(self, x, length_x=None):
        # WRITE YOUR CODE HERE
        
        x = torch.transpose(x, 0, 1)
        
        output, conv_vec = self.lstm(x, self._get_init_states(x), length_x)
        
        output = self.ln_f(conv_vec[-1][0])
        output = self.classifier_head(output)
        
        return output


class LSTMModule(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lr = self.config["lr"]
        self.LSTM = LSTMEncoder(self.config['LSTM'])
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.config['LSTM']['n_class'], average=None)
        self.F1 = torchmetrics.F1Score(task="multiclass", num_classes=self.config['LSTM']['n_class'])
        self.save_hyperparameters()
        
    def forward(self, inputs):
        return self.LSTM(inputs)
    
    def training_step(self, batch, batch_idx):
        # can i get y values here? if so, it can compute ROI 
        inputs, target = batch
        output = self.LSTM(inputs)
        loss = torch.nn.functional.cross_entropy(output, target)
        
        prob = torch.nn.functional.softmax(output, dim=1)
        accuracy = self.accuracy(prob, target)
        F1 = self.F1(prob, target)
        
        self.log_dict({"loss":loss, "accuracy_0":accuracy[0], "accuracy_1":accuracy[1], f"accuracy_2":accuracy[2], "F1":F1})
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.LSTM(inputs)
        val_loss = torch.nn.functional.cross_entropy(output, target)
        
        prob = torch.nn.functional.softmax(output, dim=1)
        val_accuracy = self.accuracy(prob, target)
        val_F1 = self.F1(prob, target)
        
        self.log_dict({"validation_loss":val_loss, "validation_accuracy_0":val_accuracy[0], "validation_accuracy_1":val_accuracy[1], "validation_accuracy_2":val_accuracy[2], "val_F1":val_F1}, sync_dist=True)
        return val_loss, val_accuracy
    
    def test_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.LSTM(inputs)
        test_loss = torch.nn.functional.cross_entropy(output, target)
        
        prob = torch.nn.functional.softmax(output, dim=1)
        test_accuracy = self.accuracy(prob, target)
        test_F1 = self.F1(prob, target)
        
        self.log_dict({"test_loss":test_loss, "test_accuracy_0":test_accuracy[0], "test_accuracy_1":test_accuracy[1], "test_accuracy_2":test_accuracy[2],  "test_F1":test_F1}, sync_dist=True)
        
        return output
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        
        inputs, ind = batch
        
        #outputs = self.LSTM(inputs)
        
        #outputs = outputs.view(3)
        
        #outputs = 0 if outputs[0] > 0.85 else 2 if outputs[2] > 0.85 else 1
        
        outputs = torch.argmax(self.LSTM(inputs), dim=1) 
        return ind, outputs
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.LSTM.parameters(), lr=self.lr)
        
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
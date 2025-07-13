import torch
import torch.nn as nn
import torch.nn.functional as F

from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)


class PPOxLSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, dropout=0):
    super().__init__()
    self.input_size = input_size

    cfg = xLSTMBlockStackConfig(
      mlstm_block=mLSTMBlockConfig(
          mlstm=mLSTMLayerConfig(
              conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
          )
      ),
      slstm_block=sLSTMBlockConfig(
          slstm=sLSTMLayerConfig(
              backend="cuda",
              num_heads=4,
              conv1d_kernel_size=4,
              bias_init="powerlaw_blockdependent",
              dropout=dropout
          ),
          feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
      ),
      context_length=256,
      num_blocks=num_layers,
      embedding_dim=hidden_size,
      slstm_at=[],
      dropout=dropout

    )
    self.num_layers = num_layers
    self.input_proj = nn.Linear(input_size, hidden_size)
    self.xlstm = xLSTMBlockStack(cfg)

    self.hidden_size = hidden_size * hidden_size + hidden_size * 2 + 4  

  def preprocess(self, seq, state=None):
    if(state is not None):
      mlstm_state = state[0]
      mlstm_state = self.preprocess_mlstm_state(mlstm_state)

      conv_state = state[1]
      conv_state = self.preprocess_conv_state(conv_state)
      state = {
        f"block_0": {
          "mlstm_state": mlstm_state,
          "conv_state": conv_state,
        }
      }

    # n_seq x bs x seq_dim -> bs x n_seq x seq_dim
    seq = torch.transpose(seq, 0, 1)
    return seq, state

  def preprocess_mlstm_state(self, mlstm_state):
    mlstm_state = mlstm_state.squeeze(0)

    start = 0
    stop = self.ks*self.H*self.H
    mlstm_state0 = mlstm_state[:, start: stop] 

    start = stop
    stop +=  self.ks*self.H
    mlstm_state1 = mlstm_state[:, start: stop] 

    start = stop
    stop +=  self.ks
    mlstm_state2 = mlstm_state[:, start: stop] 

    mlstm_state0 = mlstm_state0.reshape(self.bs, self.ks, self.H, self.H)
    mlstm_state1 = mlstm_state1.reshape(self.bs, self.ks, self.H, 1)
    mlstm_state2 = mlstm_state2.reshape(self.bs, self.ks, 1, 1)



    return (mlstm_state0, mlstm_state1, mlstm_state2)

  def preprocess_conv_state(self, conv_state):
    conv_state = conv_state.squeeze(0)
    conv_state = conv_state[:, :-self.pad_amount]
    conv_state = conv_state.reshape(self.bs, self.ks, -1)
    return (conv_state, )

  def forward(self, seq, state=None):
    try:
      print(self.ks)
    except:
      state = None
    seq, state = self.preprocess(seq, state)

    seq = self.input_proj(seq)
    seq, state = self.xlstm.step(seq, state)

    seq, state = self.postprocess(seq, state)
    return seq, state
  
  def postprocess(self, seq, state):
    block = state[f"block_0"]
    mlstm_state = block["mlstm_state"]
    conv_state = block["conv_state"]
    
    mlstm_state = self.postprocess_mlstm_state(mlstm_state)
    conv_state = self.postprocess_conv_state(conv_state)

    self.pad_amount = mlstm_state.shape[-1] - conv_state.shape[-1]
    conv_state= F.pad(conv_state, pad=(0, self.pad_amount), mode='constant', value=0)

    state = (mlstm_state, conv_state)
    seq = torch.transpose(seq, 0, 1)
    return seq, state
  
  def postprocess_mlstm_state(self, mlstm_state):
    mlstm_state0 = mlstm_state[0] # bs x ks x hidden_size/2 x hidden_size/2
    mlstm_state1 = mlstm_state[1] # bs x ks x hidden_size/2 x 1
    mlstm_state2 = mlstm_state[2] # bs x ks x 1 x 1 

    self.bs, self.ks, self.H, self.H = mlstm_state0.shape
    mlstm_state0 = mlstm_state0.reshape(self.bs, -1)
    mlstm_state1 = mlstm_state1.reshape(self.bs, -1)
    mlstm_state2 = mlstm_state2.reshape(self.bs, -1)

    mlstm_state = torch.concat([mlstm_state0, mlstm_state1, mlstm_state2], dim=-1).unsqueeze(0)
    return mlstm_state
  
  def postprocess_conv_state(self, conv_state):
    conv_state = conv_state[0].reshape(self.bs, -1).unsqueeze(0)

    return conv_state






def main():
  kwargs = {
    "dropout": 0.1
  }
  xlstm = PPOxLSTM(input_size=31, hidden_size=256, num_layers=1, **kwargs)

  n_seq = 1
  bs = 2
  seq_dim = 31


  inpu_seq = torch.randn([n_seq, bs, seq_dim])
  # print(inpu_seq.shape)
  seq, state = xlstm(inpu_seq)

  # print(seq.shape)
  # print(state[0][0].shape)
  # print(state[0][1].shape)
  # print(state[0][2].shape)
  # print(state[1][0].shape)

  seq, state = xlstm(inpu_seq, state)

  # print(seq.shape)
  # print(state[0].shape)
  # print(state[1].shape)




  lstm = nn.LSTM(input_size=31, hidden_size=256, num_layers=1, **kwargs)
  seq, state = lstm(inpu_seq)
  print(seq.shape)
  print(state[0].shape)
  print(state[1].shape)


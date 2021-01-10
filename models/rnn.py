from utils.utils import *

import torch
import torch.optim as optim

def make_coder(player,use_skip,search,hidden_size):# 1.4.1確認済
    if use_skip and player=='Gen':
        ij_to_k_encoder = {
            (0,0):0, (0,2):1, (0,4):2, (0,6):3,
            (1,0):4, (1,2):1, (1,4):2, (1,6):3,
            (2,0):5, (2,2):1, (2,4):2, (2,6):3,
        }
        encoders = nn.ModuleList([
            nn.Identity(),
            nn.Linear(len(search[1 ]),hidden_size),
            nn.Linear(len(search[3 ]),hidden_size),
            nn.Linear(len(search[5 ]),hidden_size),
            nn.Linear(len(search[7 ]),hidden_size),
            nn.Linear(len(search[15]),hidden_size),
        ])
        ij_to_k_decoder = {
            (0,0):0, (0,2):1, (0,4):2, (0,6):3,
            (1,0):0, (1,2):1, (1,4):2, (1,6):4,
            (2,0):0, (2,2):1, (2,4):2, (2,6):5,
        }
        decoders = nn.ModuleList([
            nn.Linear(hidden_size,len(search[0 ])),
            nn.Linear(hidden_size,len(search[2 ])),
            nn.Linear(hidden_size,len(search[4 ])),
            nn.Linear(hidden_size,len(search[6 ])),
            nn.Linear(hidden_size,len(search[14])),
            nn.Linear(hidden_size,len(search[22])),
        ])
    elif use_skip and player=='Dis':
        ij_to_k_encoder = {
            (0,1):0, (0,3):1, (0,5):2, (0,7):3,
            (1,1):0, (1,3):1, (1,5):2, (1,7):4,
            (2,1):0, (2,3):1, (2,5):2, (2,7):5,
        }
        encoders = nn.ModuleList([
            nn.Linear(len(search[0 ]),hidden_size),
            nn.Linear(len(search[2 ]),hidden_size),
            nn.Linear(len(search[4 ]),hidden_size),
            nn.Linear(len(search[6 ]),hidden_size),
            nn.Linear(len(search[14]),hidden_size),
            nn.Linear(len(search[22]),hidden_size),
        ])
        ij_to_k_decoder = {
            (0,1):0, (0,3):1, (0,5):2, (0,7):3,
            (1,1):0, (1,3):1, (1,5):2, (1,7):4,
            (2,1):0, (2,3):1, (2,5):2, (2,7):5,
        }
        decoders = nn.ModuleList([
            nn.Linear(hidden_size,len(search[1 ])),
            nn.Linear(hidden_size,len(search[3 ])),
            nn.Linear(hidden_size,len(search[5 ])),
            nn.Linear(hidden_size,len(search[7 ])),
            nn.Linear(hidden_size,len(search[15])),
            nn.Linear(hidden_size,len(search[23])),
        ])
    elif (not use_skip) and player=='Gen':
        ij_to_k_encoder = {
            (0,0):0, (0,2):1, (0,4):2,
            (1,0):3, (1,2):1, (1,4):2,
            (2,0):3, (2,2):1, (2,4):2,
        }
        encoders = nn.ModuleList([
            nn.Identity(),
            nn.Linear(len(search[1 ]),hidden_size),
            nn.Linear(len(search[3 ]),hidden_size),
            nn.Linear(len(search[5 ]),hidden_size),
        ])
        ij_to_k_decoder = {
            (0,0):0, (0,2):1, (0,4):2,
            (1,0):0, (1,2):1, (1,4):2,
            (2,0):0, (2,2):1, (2,4):2,
        }
        decoders = nn.ModuleList([
            nn.Linear(hidden_size,len(search[0 ])),
            nn.Linear(hidden_size,len(search[2 ])),
            nn.Linear(hidden_size,len(search[4 ])),
        ])
    elif (not use_skip) and player=='Dis':
        ij_to_k_encoder = {
            (0,1):0, (0,3):1, (0,5):2,
            (1,1):0, (1,3):1, (1,5):2,
            (2,1):0, (2,3):1, (2,5):2,
        }
        encoders = nn.ModuleList([
            nn.Linear(len(search[0 ]),hidden_size),
            nn.Linear(len(search[2 ]),hidden_size),
            nn.Linear(len(search[4 ]),hidden_size),
        ])
        ij_to_k_decoder = {
            (0,1):0, (0,3):1, (0,5):2,
            (1,1):0, (1,3):1, (1,5):2,
            (2,1):0, (2,3):1, (2,5):2,
        }
        decoders = nn.ModuleList([
            nn.Linear(hidden_size,len(search[1 ])),
            nn.Linear(hidden_size,len(search[3 ])),
            nn.Linear(hidden_size,len(search[5 ])),
        ])
    return ij_to_k_encoder,encoders,ij_to_k_decoder,decoders

class RNN_Net(nn.Module):# 1.4.1確認済
    def __init__(self,option,player):
        super(RNN_Net,self).__init__()
        self.rnn = nn.LSTMCell(option.hidden_size,option.hidden_size)
        self.ij_to_k_encoder,self.encoders,self.ij_to_k_decoder,self.decoders = \
            make_coder(player,option.use_skip,option.search,option.hidden_size)
        
    def forward(self,x,hidden,i_stage,j_move):# 1.4.3確認済
        h0 = self.encoders[self.ij_to_k_encoder[(i_stage,j_move)]](x)
        h1, c1 = self.rnn(h0,hidden)
        z = self.decoders[self.ij_to_k_decoder[(i_stage,j_move)]](h1)
        return z, (h1.detach(),c1.detach())

class RNN_Model():# 1.4.1確認済
    '''playerの勝率を最大化する方策を学習'''
    def __init__(self,option,player):
        assert (player=='Gen' or player=='Dis'),'RNN_Netの引数エラー'
        self.player = player
        self.net = RNN_Net(option,player).to(option.device)
        self.net.apply(weights_init)
        self.optimizer = optim.Adam(self.net.parameters(),lr=option.lrC,betas=(option.beta1,option.beta2))
        self.rewards_window = Window(option.reward_window_size)

    def train(self,winner,log_probs):
        reward = 1 if self.player == winner else -1
        self.rewards_window.push(reward)
        baseline = self.rewards_window.mean if self.rewards_window.is_full() else 0
        adv = reward - baseline

        # policy loss
        log_probs = torch.cat(log_probs, -1)
        loss = -log_probs * adv
        loss = loss.sum()

        # update controller
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
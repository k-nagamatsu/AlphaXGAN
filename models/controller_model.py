from utils.utils import *
from models import monte_carlo_tree, rnn

import torch
import torch.nn.functional as F

class Controller_Model():
    def __init__(self,option):# 1.4.1確認済
        self.option = option
        self.tree = monte_carlo_tree.Tree(option)
        self.ctrlG = rnn.RNN_Model(option,player='Gen') if option.use_controllerG else None
        self.ctrlD = rnn.RNN_Model(option,player='Dis') if option.use_controllerD else None

    def init_hidden(self):# 1.4.3確認済
        return torch.zeros(1,self.option.hidden_size,requires_grad=False).to(self.option.device)

    def sample(self,stage):# 1.4.6確認済
        '''
        designをサンプリングする
        Return
        ______
        design : list
            design[i_stage][j_move] = child_modelの構造(skip以外:str, skip:tuple)
        node_id : int
            プレイアウトを始めたnode = backupの起点
        '''
        x = self.init_hidden()
        hidden = (self.init_hidden(), self.init_hidden())
        design = [[None for _ in range(self.option.N_move)] for _ in range(stage)]
        log_probs = [[],[]]

        node_id = 0
        depth = 0 # rootからの深さ
        mode = 'select' # 'select' or 'expand' or 'rollout'
        for i_stage in range(stage):
            for j_move in range(self.option.N_move):
                ctrl = self.ctrlG if j_move % 2 == 0 else self.ctrlD
                if mode == 'select':
                    if ctrl:# モンテカルロ木探索にctrlを使用する
                        z, hidden = ctrl.net(x,hidden,i_stage,j_move)
                        log_prob = F.log_softmax(z, dim=-1)
                        rnn_policy = F.softmax(z,dim=1).view(-1).detach().to('cpu').numpy()
                        log_probs[j_move % 2].append(log_prob)
                    else:
                        rnn_policy = 1
                    node_id,action,x = self.tree.select(node_id,rnn_policy,depth)
                    if self.tree.nodes[node_id].childlen == None:
                        mode = 'expand'
                elif mode == 'expand':
                    self.tree.expand(node_id,depth)
                    mode = 'rollout'
                if mode == 'rollout': # ミスではない
                    if self.option.use_controller_rollout:
                        with torch.no_grad():
                            z, hidden = ctrl.net(x,hidden,i_stage,j_move)
                            rnn_policy = F.softmax(z,dim=1).view(-1).detach().to('cpu').numpy()
                        action,x = self.tree.rollout(depth,rnn_policy)
                    else:
                        action = self.tree.rollout(depth)
                design[i_stage][j_move] = action
                depth += 1

        return design,log_probs,node_id
    
    def train(self,testloader,stage,search_iter,child_model):# 1.4.1確認済
        if self.option.use_controllerG: self.ctrlG.net.train()
        if self.option.use_controllerD: self.ctrlD.net.train()
        N_win_G = 0
        N_win_D = 0

        for epoch in range(self.option.N_epoch_SDTR):
            design,log_probs,last_node_id = self.sample(stage)
            winner,TPTN = child_model.fight(design,testloader)
            if self.option.use_controllerG and log_probs[0] != []: self.ctrlG.train(winner,log_probs[0])
            if self.option.use_controllerD and log_probs[1] != []: self.ctrlD.train(winner,log_probs[1])

            # update tree
            self.tree.backup(last_node_id,winner)

            if winner == 'Gen': N_win_G += 1
            if winner == 'Dis': N_win_D += 1

        text =  f'{search_iter},{N_win_G},{N_win_D}'
        print(f'SDTR,{text}')
        write(f'../output/log/SDTR.csv',text+'\n')
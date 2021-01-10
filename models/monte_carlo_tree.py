from utils.utils import *

import random
from collections import deque
import numpy as np
import networkx as nx

class Node():# 1.4.1確認済
    def __init__(self,player,parent):
        self.player   = player # その手のプレイヤー
        self.parent   = parent # 親ノードid
        self.childlen = None   # 子ノードidのリスト(子ノード未作成==None)
        self.win      = 0      # そのノードplayerの勝利回数
        self.trial    = 0      # そのノードの試行回数

    def get_playout_value(self,nodes):
        '''子ノードのplayoutの評価値を返す関数'''
        return np.array([nodes[node_id].win / (nodes[node_id].trial + 1) for node_id in self.childlen])

    def get_bias_value(self,nodes):
        '''子ノードのbiasの評価値を返す関数'''
        trials = np.array([nodes[node_id].trial for node_id in self.childlen])
        return np.sqrt(np.sum(trials)) / (1 + trials)

class Tree():# 1.4.1確認済
    '''探索空間のグラフ構造'''
    def __init__(self,option):
        '''
        self.nodes : list
            self.nodes[node_id] : Node()
        '''
        self.option = option
        depth=0 # 0:root, 1:Gen, 2:Dis, 3:Gen, 4:Dis, ... 
        node_id = 0
        self.nodes = [Node(player='root',parent=None)]
        childlen = list(range(len(self.nodes),len(self.nodes)+len(option.search[depth])))
        self.nodes[node_id].childlen = childlen
        self.nodes += [Node(player='Gen',parent=node_id) for _ in childlen]
        self.onehot = [torch.eye(len(types)) for types in option.search]

    def save_nodes(self,path):
        dump_pickle(self.nodes,path)

    def select(self,node_id,rnn_policy,depth):
        '''
        方策に基づきnode_idの子ノードの中から 次のノード=行動 を選択する関数
        __________
        Parameters
        __________
        rnn_policy : np.array or int
            np.array → ctrlが出力した確率方策
            1        → ctrlをモンテカルロ木探索に使用しない
        __________
        Return
        __________
        next_node_id : int
            node_idの子ノードのうち、次に選ぶノード
        action : str
            相手playerの次の行動(構造)の略称
        '''
        node = self.nodes[node_id]
        playout_value = node.get_playout_value(self.nodes)
        bias_value = node.get_bias_value(self.nodes)
        policy = playout_value + self.option.rho * rnn_policy * bias_value
        index_action = np.argmax(policy)
        next_node_id = node.childlen[index_action]
        action = self.option.search[depth][index_action]
        x = self.onehot[depth][index_action].view(1,-1).to(self.option.device)
        return next_node_id,action,x

    def expand(self,node_id,depth):
        '''リーフノードの試行回数が閾値以上のとき、子ノードの展開'''
        if self.nodes[node_id].trial > self.option.trial_threshold:
            childlen = list(range(len(self.nodes),len(self.nodes)+len(self.option.search[depth])))
            self.nodes[node_id].childlen = childlen
            player = 'Dis' if depth % 2 == 0 else 'Gen'
            self.nodes += [Node(player=player,parent=node_id) for _ in childlen]
            
    def rollout(self,depth,rnn_policy=None):
        '''
        node_idの子ノードの中からランダムに選ぶロールアウトを実施
        __________
        Return
        __________
        action : str
            相手playerの次の行動(構造)の略称
        '''
        childlen = self.option.search[depth]
        index_list = list(range(len(childlen)))
        index_action = np.random.choice(index_list, p=rnn_policy) if type(rnn_policy) == np.ndarray else np.random.choice(index_list)
        action = childlen[index_action]
        if type(rnn_policy) == np.ndarray:
            x = self.onehot[depth][index_action].view(1,-1).to(self.option.device)
            return action,x
        else:
            return action

    def backup(self,node_id,winner):
        '''リーフノードからルートノードへ結果を反映'''
        while self.nodes[node_id].parent != None:
            self.nodes[node_id].trial += 1
            if self.nodes[node_id].player == winner:
                self.nodes[node_id].win += 1
            node_id = self.nodes[node_id].parent
        self.nodes[node_id].trial += 1

    def get_best_design(self,stage):
        '''試行回数最大のdesignを取得'''
        design = [[] for _ in range(stage)]
        trials = []
        node_id = 0
        for i_stage in range(len(design)):
            for j_move in range(self.option.N_move):
                if node_id != None:
                    if self.nodes[node_id].childlen == None:
                        node_id = None
                    else:
                        index = np.argmax([self.nodes[child_id].trial for child_id in self.nodes[node_id].childlen])
                        node_id = self.nodes[node_id].childlen[index]
                        design[i_stage].append(self.option.search[i_stage * self.option.N_move + j_move][index])
                        trials.append(self.nodes[node_id].trial)
                if node_id == None:# ミスではない
                    childlen = self.option.search[i_stage * self.option.N_move + j_move]
                    index = np.random.randint(0,len(childlen))
                    design[i_stage].append(childlen[index])
                    trials.append('None')

        return design,trials

    def convert_networkx(self):
        '''treeを深さ優先探索でnetworkXのGに変える'''
        threshold = 0 # 閾値より大きい試行回数のみplot
        G = nx.DiGraph()
        s = []
        for v in self.nodes[0].childlen:
            if self.nodes[v].trial > threshold:
                s.append((0,v))
        while len(s) != 0:
            u,v = s.pop()
            G.add_edge(u,v)
            G.nodes[v]['WP'] = 1
            for w in self.nodes[v].childlen:
                if self.nodes[w].trial > threshold:
                    s.append((v,w))
        return G

    def plot(self):
        G = self.convert_networkx()
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        fig = plt.figure(figsize=(10, 10), dpi=300)
        ax = fig.add_subplot(1, 1, 1)
        nx.draw(G,
            pos=pos,
            arrows=False,
            with_labels=False,
            ax=ax,
            node_size=100,
            #node_color=range(len(logs)+1),
            cmap="jet",
            width=0.5,
        )
        plt.savefig('../output/tree.png')
        plt.close()
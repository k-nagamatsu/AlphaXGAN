import pickle,os,shutil
from copy import deepcopy
from datetime import datetime
import dateutil.tz
from collections import deque
import numpy as np
import torch
import torch.nn as nn

def initialize(filepath):
    os.makedirs(filepath,exist_ok=True)
    shutil.rmtree(filepath)
    os.makedirs(filepath)

def load_pickle(path):
    '''pickleの読み込み関数'''
    with open(path,'rb') as f:
        data = pickle.load(f)
    return data

def dump_pickle(data,path):
    '''pickleの書き出し関数'''
    with open(path,'wb') as f:
        pickle.dump(data,f)

def write(filepath,text):
    with open(filepath,'a') as f:
        f.write(text)

def fix_random_seed(option):
    np.random.seed(option.random_seed)
    torch.manual_seed(option.random_seed)
    torch.cuda.manual_seed(option.random_seed)

def weights_init(m):# 1.4.1確認済
    classname = m.__class__.__name__
    init_type = 'xavier_uniform'
    if classname.find('Conv2d') != -1:
        if init_type == 'normal':
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif init_type == 'orth':
            nn.init.orthogonal_(m.weight.data)
        elif init_type == 'xavier_uniform':
            nn.init.xavier_uniform_(m.weight.data, 1.)
        else:
            raise NotImplementedError('{} unknown inital type'.format(args.init_type))
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

class Window():
    '''
    window幅の移動平均と移動分散を求める関数
    (np.varより圧倒的に処理が高速)
    '''
    def __init__(self, size):
        self.size = size # window size
        self.mean = 0
        self.var = 0
        self.window = deque(maxlen=size)

    def clear(self):
        self.window.clear()
        self.mean = 0
        self.var = 0

    def is_full(self):
        return len(self.window) == self.size

    def push(self, x):
        if self.is_full():
            x_removed = self.window.popleft()
            self.window.append(x)
            old_mean = self.mean
            self.mean += (x - x_removed) / self.size
            self.var += (old_mean-self.mean)*(2*(self.size*self.mean-x)-(self.size-1)*(x+x_removed))/ self.size
        else:
            old_mean = self.mean
            self.mean = (self.mean*len(self.window)+x)/(len(self.window)+1)
            self.var = (self.var*len(self.window)+(len(self.window)*(old_mean-self.mean)**2)+(x-self.mean)**2)\
                /(len(self.window)+1)
            self.window.append(x)

def set_log_dir(root_dir, exp_name):#1.4.1確認済
    path_dict = {}
    os.makedirs(root_dir, exist_ok=True)

    # set log path
    exp_path = os.path.join(root_dir, exp_name)
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    prefix = exp_path + '_' + timestamp
    os.makedirs(prefix)
    path_dict['prefix'] = prefix

    # set checkpoint path
    ckpt_path = os.path.join(prefix, 'Model')
    os.makedirs(ckpt_path)
    path_dict['ckpt_path'] = ckpt_path

    log_path = os.path.join(prefix, 'Log')
    os.makedirs(log_path)
    path_dict['log_path'] = log_path

    # set sample image path for fid calculation
    sample_path = os.path.join(prefix, 'Samples')
    os.makedirs(sample_path)
    path_dict['sample_path'] = sample_path

    return path_dict

def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)

def copy_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten
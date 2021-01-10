import parser
from utils.utils import *
from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception
from utils import dataset
from models.controller_model import Controller_Model
from models.child_model import Child_Model

import time,os,itertools

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def search_design(option):# 1.4.4確認済
    '''最適な構造を探索する'''
    initialize('../output')
    os.makedirs('../output/log')
    os.makedirs('../output/img')
    os.makedirs('../output/model')

    fix_random_seed(option)
    _init_inception()
    inception_path = check_or_download_inception(None)
    create_inception_graph(inception_path)

    stage = 1
    trainloader,testloader,_ = dataset.make_dataset(option,img_size=2**(stage + 2))
    child_model = Child_Model(option)
    controller_model = Controller_Model(option)

    design_to_score = {}
    time_all = 0
    print('start search design')
    for search_iter in range(option.max_search_iter):
        time_all_start = time.time()
        if search_iter in  option.grow_step:
            stage += 1
            trainloader,testloader,_ = dataset.make_dataset(option,img_size=2**(stage + 2))
            if option.reset_controller:
                del controller_model
                controller_model = Controller_Model(option)
        
        start = time.time()
        flag_reset = child_model.train('SDTG',trainloader,stage,search_iter,design=None,controller_model=controller_model)
        time_SDTG = int(time.time() - start)+1

        start = time.time()
        controller_model.train(testloader,stage,search_iter,child_model)
        time_SDTR = int(time.time() - start)+1
        time_all += int(time.time() - time_all_start)+1

        if flag_reset:
            del child_model
            child_model = Child_Model(option)

        start = time.time()
        design,trials = controller_model.tree.get_best_design(stage)
        key = list(itertools.chain.from_iterable(design))
        key = '-'.join(map(str, key))
        text = f'{search_iter},{trials},{key}\n'
        write('../output/log/search_iter_best_design.csv',text)

        if (trials[-1] != 'None') and (key not in design_to_score.keys()):
            text = f'{search_iter},{key}\n'
            write('../output/log/no_rollout_design.csv',text)

        if key in design_to_score.keys():
            FID, mean, std = design_to_score[key]
            flag_dict = True
        else:
            # designを0から学習し、生成した画像を評価する
            fixed_child_model = Child_Model(option)
            FID, mean, std, _, _ = fixed_child_model.train('EDTG',trainloader,stage,search_iter,design=design)
            design_to_score[key] = (FID, mean, std)
            del fixed_child_model
            flag_dict = False

        text =  f'{search_iter},{time_all},{FID},{mean},{std},{flag_dict}\n'
        write(f'../output/log/EDTG_iter_score.csv',text)

        time_ED = int(time.time() - start)+1

        text = f'{search_iter},{time_SDTG},{time_SDTR},{time_ED},{time_all},{flag_reset}\n'
        write('../output/log/search_iter_time.csv',text)
        if 60*60*24 < time_all:
            break

if __name__ == '__main__':
    option = parser.get_args()
    search_design(option)
    print('finish!')
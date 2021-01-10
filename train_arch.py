import parser
from utils.utils import *
from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception
from utils import dataset
from models.child_model import Child_Model

import os

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def train_design(option):
    initialize('../output')
    os.makedirs('../output/log')
    os.makedirs('../output/img')
    os.makedirs('../output/model')

    fix_random_seed(option)
    _init_inception()
    inception_path = check_or_download_inception('./tmp/imagenet/')
    create_inception_graph(inception_path)

    stage = 3
    trainloader,_,_ = dataset.make_dataset(option,img_size=2**(stage + 2))

    design14 = [
        ['deconv','conv_3x3_2','conv_5x5_1','conv_5x5_1','conv_5x5_1','conv_5x5_2',(0,),(0,)],
        ['deconv','max','conv_5x5_1','conv_5x5_2','conv_3x3_1','conv_3x3_1',(0, 1),(1, 1)],
        ['deconv','conv_5x5_2','conv_5x5_1','conv_5x5_2','conv_3x3_1','conv_1x1_1',(0, 0, 0),(1, 0, 0)]
    ]

    design31 = [
        ['deconv','max','conv_1x1_1','conv_5x5_1','conv_3x3_1','conv_3x3_1',(1,),(1,)],
        ['deconv','conv_3x3_2','conv_5x5_1','conv_3x3_2','conv_3x3_1','conv_3x3_1',(1, 1),(1, 1)],
        ['nearest','max','conv_5x5_1','conv_3x3_2','conv_1x1_1','conv_3x3_1',(0, 0, 0),(0, 0, 1)]
    ]
    design = design31

    child_model = Child_Model(option)
    _, _, _, best_FID_epoch, best_mean_epoch = child_model.train('EDTG',trainloader,stage,0,design=design)

    del child_model
    option.num_eval_imgs = 50000
    child_model = Child_Model(option)
    child_model.netG.to('cpu')
    child_model.netG.load_state_dict(torch.load(f'../output/model/netG_epoch{best_mean_epoch}'))
    child_model.netG.to(option.device)
    mean,std,fid_score = child_model.test(design,-1,-1,calculate_FID=True)
    text =  f'{mean},{std},{fid_score},{best_FID_epoch}, {best_mean_epoch},{design}\n'
    write(f'../output/log/best_design_score.csv',text)

if __name__ == '__main__':
    option = parser.get_args()
    train_design(option)
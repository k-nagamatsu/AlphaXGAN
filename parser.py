from utils.utils import *

import argparse
import torch

def get_args():#1.4.1確認済
    parser = argparse.ArgumentParser()

    # 実行に関する設定
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--run_mode', type=str, choices=['debug','normal'], default='debug')

    # データに関する設定
    parser.add_argument('--dataset', type=str, choices=['cifar10','stl10'], default='cifar10',
                        help='dataset type')
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='dimensionality of the latent space')
    parser.add_argument('--gf_dim', type=int, default=128,
                        help='The base channel num of gen')
    parser.add_argument('--df_dim', type=int, default=64,
                        help='The base channel num of disc')
    parser.add_argument('--bottom_width', type=int, default=4,
                        help="the base resolution of the GAN")
    parser.add_argument('--num_eval_imgs', type=int, default=500)

    # 学習に関する一般的設定
    parser.add_argument('--lrG', type=float, default=0.0002,
                        help='adam: G learning rate')
    parser.add_argument('--lrD', type=float, default=0.0002,
                        help='adam: D learning rate')
    parser.add_argument('--lrC', type=float, default=3.5e-4,
                        help='adam: C learning rate')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='adam: decay of first order momentum of gradient')
    parser.add_argument('--beta2', type=float, default=0.9,
                        help='adam: decay of first order momentum of gradient')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of cpu threads to use during batch generation')
    parser.add_argument('--grow_step1', type=int, default=5 ,
                        help='which iteration to grow the image size from 8 to 16')
    parser.add_argument('--grow_step2', type=int, default=10,
                        help='which iteration to grow the image size from 16 to 32')
    parser.add_argument('--max_search_iter', type=int, default=200,
                        help='max search iterations of this algorithm')

    # child_modelの学習に関する設定
    parser.add_argument('--use_skip',action='store_true')
    parser.add_argument('--loss', type=str, choices=['Hinge','BCE'])
    parser.add_argument('--N_epoch_SDTG', type=int, default=15,
                        help='the number of epoch to train the gan at each search iteration')
    parser.add_argument('--N_epoch_EDTG', type=int, default=20,#90,
                        help='the number of epoch to train the gan at each search iteration')
    parser.add_argument('--train_batch_size', type=int, default=64,
                        help='size of the batches')
    parser.add_argument( '--test_batch_size', type=int, default=100,
                        help='size of the batches')
    parser.add_argument('--fixed_noise_size', type=int, default=100,
                        help='size of fixed_noise')
    parser.add_argument('--fight_noise_size', type=int, default=100,
                        help='size of fight_noise')
    parser.add_argument('--use_dynamic_reset',action='store_true')
    parser.add_argument('--dynamic_reset_window_size', type=int, default=500,
                        help='the window size')
    parser.add_argument('--dynamic_reset_threshold', type=float, default=1e-4,
                        help='var threshold')

    # controllerの学習に関する設定
    parser.add_argument('--use_controllerG',action='store_true')
    parser.add_argument('--use_controllerD',action='store_true')
    parser.add_argument('--reset_controller',action='store_true')
    parser.add_argument('--N_epoch_SDTR', type=int, default=300,#30,
                        help='number of steps to train the controller at each search iteration')
    parser.add_argument('--hidden_size', type=int, default=100,
                        help='the size of hidden vector')
    parser.add_argument('--reward_window_size', type=int, default=500,
                        help='the window size')

    # モンテカルロ木探索に関する設定
    parser.add_argument('--rho', type=float, default=0.5,
                        help='UCT1 param')
    parser.add_argument('--trial_threshold', type=int, default=5,
                        help='trial threshold')
    parser.add_argument('--use_controller_rollout',action='store_true')

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    assert (args.loss in ['Hinge','BCE'])

    # 探索空間に関する設定
    NORMALs = ('conv_1x1_1','conv_3x3_1','conv_5x5_1','conv_3x3_2','conv_5x5_2')
    UPs = ('deconv','nearest','bilinear')
    DOWNs = ('ave','max','conv_3x3_1','conv_5x5_1','conv_3x3_2','conv_5x5_2')
    SKIPs = ( # (そのCellの最後にCell1のinputをスキップ入力するかどうか,...)
        ((0,), (1,)), # Cell1
        ((0,0), (0,1), (1,0), (1,1)), # Cell2
        ((0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)) # Cell3
    )
    tmp = [UPs,DOWNs,NORMALs,NORMALs,NORMALs,NORMALs]
    tmp_skip = tmp + [SKIPs[0]] + [SKIPs[0]] + tmp + [SKIPs[1]] + [SKIPs[1]] + tmp + [SKIPs[2]] + [SKIPs[2]]
    tmp_arch = tmp + tmp + tmp
    args.search = tuple(tmp_skip) if args.use_skip else tuple(tmp_arch)
    args.N_move = len(args.search)//3 # 6 or 8
    args.N_arch = len(tmp) # 6
    args.NORMALs = NORMALs
    args.UPs = UPs
    args.DOWNs = DOWNs
    args.SKIPs = SKIPs
    args.grow_step = (args.grow_step1,args.grow_step2)

    # デバッグモード設定
    if args.run_mode == 'debug':
        args.train_batch_size = 10
        args.test_batch_size = 10
        args.dynamic_reset_window_size = 10
        args.N_epoch_SDTG = 2
        args.N_epoch_EDTG = 2
        args.N_epoch_SDTR = 2
        args.max_search_iter = 4
        args.trial_threshold = 1
        args.grow_step = (1,2)
        args.num_eval_imgs = 10

    # score計算用
    args.path_helper = set_log_dir('logs', 'alphaxgan_search')

    return args
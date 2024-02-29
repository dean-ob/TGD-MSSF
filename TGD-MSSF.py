import argparse
import os
import torch
from utils.exp.exp_tgdmssf import Exp_TGDMSSF
# from utils_our_s.exp.exp_preformer import My_loss
import random
import numpy as np


def main():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='our & Transformer family for Time Series Forecasting')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model', type=str, required=True, default='tgd_mssf',
                        help='model name, options: [tgd_mssf]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='Price_WTI', help='data, '
                        'options : [Prcie_Data]')
    parser.add_argument('--root_path', type=str, default='F:\学术硕士学位申请表格\新建文件夹\第二章模型\data/',
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='price_data.xls', help='data file')
    parser.add_argument('--features', type=str, default='S',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='Ethylene', help='target feature in S or MS task, '
                        'Ethylene, Crude Oil, Naphtha, Carbinol, Butane, Propane, Ethylene Glycol, Ethylene Oxide, '
                                                                       'PVC, EVA, HDPE, LDPE')
    parser.add_argument('--freq', type=str, default='d',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=20, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=10, help='start token length')
    parser.add_argument('--pred_len', type=int, default=20, help='prediction sequence length')

    # model define
    parser.add_argument('--a', type=int, default=[70], help='encoder input size')
    parser.add_argument('--b', type=int, default=[850], help='encoder input size')
    parser.add_argument('--R', type=float, default=[5400], help='encoder input size')
    parser.add_argument('--Rsize', type=float, default=1, help='fenweishu')
    parser.add_argument('--q', type=float, default=0.95, help='fenweishu')
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
    parser.add_argument('--d_model_tgd', type=int, default=64, help='dimension of model')
    parser.add_argument('--d_q', type=int, default=8, help='dimension of query')
    parser.add_argument('--d_k', type=int, default=8, help='dimension of key')
    parser.add_argument('--d_v', type=int, default=8, help='dimension of value')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=4, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=5, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--padding', type=int, default=0, help='padding type')
    parser.add_argument('--enc_L0', type=int, default=2, help='L0')
    parser.add_argument('--dec_L0', type=int, default=2, help='L0')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')

    # optimization
    parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=5, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_TGDMSSF

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = ('_{}_{}_ft{}_lr{}_Rs{}_co{}_eL0{}_dL0{}_sl{}_ll{}_pl{}_dm{}_dmt{}_nh{}_dq{}_dk{}_dv{}_el{}_dl{}'
                       '_df{}_fc{}_eb{}_do{}_tg{}_{}_{}').format(
                args.model,
                args.data,
                args.features,
                args.learning_rate,
                args.Rsize,
                args.c_out,
                args.enc_L0,
                args.dec_L0,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.d_model_tgd,
                args.n_heads,
                args.d_q,
                args.d_k,
                args.d_v,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.dropout,
                args.target,
                args.des, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = ('_{}_{}_ft{}_lr{}_Rs{}_co{}_eL0{}_dL0{}_sl{}_ll{}_pl{}_dm{}_dmt{}_nh{}_dq{}_dk{}_dv{}_el{}_dl{}'
                   '_df{}_fc{}_eb{}_do{}_tg{}_{}_{}').format(
            args.model,
            args.data,
            args.features,
            args.learning_rate,
            args.Rsize,
            args.c_out,
            args.enc_L0,
            args.dec_L0,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.d_model_tgd,
            args.n_heads,
            args.d_q,
            args.d_k,
            args.d_v,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.dropout,
            args.target,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()


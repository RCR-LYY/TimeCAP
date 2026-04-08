import argparse

def my_list(string):
    if isinstance(string, str):
        return [int(i) for i in string.strip('[').strip(']').split(',')]
    else:
        return string

def get_args():
    parser = argparse.ArgumentParser(description='TimeCAP_args')

    parser.add_argument('--learning_rate', type=float, default=3.7612881929789416e-05)
    parser.add_argument("--lr_decay", type=float, default=0.3, help="learning rate decay")  # 衰减系数
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout')

    parser.add_argument('--pretrain_pred_len', type=int, default=16, help='prediction sequence length for pretrain GPHT across multiple datasets [48]')
    parser.add_argument('--pretrain_batch_size', type=int, default=16, help='batch size of pre-train input data [32]')
    parser.add_argument('--d_model', type=int, default=736, help='dimension of model')
    parser.add_argument('--d_ff', type=int, default=992, help='dimension of fcn')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--depth', type=int, default=2, help='num of MultiscaleBlocks')
    parser.add_argument('--patch_len', type=my_list, default=[96, 24], help='[16]')
    parser.add_argument('--stride_time', type=my_list, default=[96, 24], help='[16]')
    parser.add_argument('--window_size', type=my_list, default=[3, 3], help='window size')
    parser.add_argument('--stride_channel', type=my_list, default=[1, 1], help='attn factor')
    parser.add_argument('--scope', type=int, default=0, help='attention time patch num')
    parser.add_argument('--optimizer', type=str, default='adam', help='choose optimizer')

    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--best_pretrain_path', type=str, default=r".\\results\\TimeCAP\\model\\finetune_TimeCAP_ETTh2_sl96\\ckpt_best.pth")
    parser.add_argument('--lambda1', type=float, default=0.8361538000800285, help='autoregressive loss coefficient')
    parser.add_argument('--lambda2', type=float, default=0.6163727742056744, help='one-shot loss coefficient')
    parser.add_argument('--lambda3', type=float, default=0.6885377212461313, help='self-distillation loss coefficient')

    parser.add_argument('--use_os_head', action='store_true', default=True)
    parser.add_argument('--use_ar_head', action='store_true', default=True)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=float, default=0.3326362081926146)

    parser.add_argument('--efficiency', help='output attention', default=False)

    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument("--lradj", type=str, default="decay", help="adjust learning rate: [constant, decay, cosine, piecewise, decay_3_0.9, decay_5_0.1, decay_10_0.1, decay_15_0.1, decay_25_0.1]")
    parser.add_argument('--pretrain_data', type=str, default='pretrain_AAAI', help='dataset type for pretrain GPHT across multiple datasets')
    parser.add_argument('--root_path_pretrain', type=str, default=r'E:/test', help='root path of the data file')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='output attention', default=False)
    parser.add_argument('--flash_attention', action='store_true', help='flash attention', default=False)
    parser.add_argument('--covariate', action='store_true', help='use cov', default=False)

    return parser
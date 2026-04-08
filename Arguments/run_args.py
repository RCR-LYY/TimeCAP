import argparse

def get_run_args():
    parser = argparse.ArgumentParser(description='run_args')

    parser.add_argument('--model', type=str, default='TimeCAP', help='model name, options: [TimeCAP]')
    parser.add_argument('--paradigm', type=str, default='pretrain_finetune', help='task paradigm, options:[pretrain_finetune, train_val_test]')
    parser.add_argument('--task_name', type=str, default='finetune', help='task name, options:[pretrain, finetune, long_term_forecast]')
    parser.add_argument('--downstream_task', type=str, default='forecasting', help='task name, options:[forecasting, imputation]')
    parser.add_argument('--is_training', type=bool, default=True, help='status')
    parser.add_argument("--load_checkpoints", type=bool, default=True, help="load historical models")
    parser.add_argument('--seed', type=int, default=2021, help='random seed')

    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    parser.add_argument('--mask_rate', type=float, default=0.125, help='mask ratio')

    parser.add_argument('--data', type=str, default='ETTh2', help='dataset type')
    parser.add_argument('--root_path', type=str, default=r"E:/Data_time_series/all_datasets/LSTF/ETT-small", help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh2.csv', help='data file')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--percent', type=int, default=100, help='Zero-shot or few-shot training')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--res_dir', type=str, default='./results', help='results dir')
    parser.add_argument('--drop_last', type=bool, default=False, help='drop the last batch')

    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--use_amp', action='store_true', default=False, help='use automatic mixed precision training')

    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False, help='use multiple gpus')
    parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')

    parser.add_argument('--use_dtw', type=bool, default=False, help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', default=False, help='inverse output data')
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")

    return parser
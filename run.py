import torch
from utils import args_fusion
from Arguments import load_setting
from exp.exp_TimeCAP import Exp_TimeCAP
from utils.tools import set_seed, make_dir, init_logger


if __name__ == '__main__':
    # Set hyperparameters
    args, _, _ = args_fusion.merge_args()
    setting = load_setting.get_setting_str(args)

    # Set random seed
    set_seed(fix_seed=args.seed)

    # Set result files path
    test_dir, model_dir, log_dir = make_dir(args)
    logger = init_logger(log_dir)

    # Set device
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))

    # Set experiment
    Exp_map = {
        'TimeCAP': Exp_TimeCAP,
    }
    Exp = Exp_map[args.model]
    exp = Exp(args, logger, model_dir, test_dir, setting)

    # Start experiment
    if args.paradigm == 'pretrain_finetune' and args.task_name == 'pretrain':
        # Start pretraining
        if args.is_training:
            # Start pretraining-training across multiple datasets
            print('>>>>>>>Start pretraining training: {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            logger.info('>>>>>>>Start pretraining training: {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            val_loss_min = exp.pretrain()

            # Start Inferencing
            print('>>>>>>>Start pretraining Inferencing: {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            logger.info('>>>>>>>Start pretraining Inferencing: {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            mse, mae = exp.Inference()

        # Start Inferencing
        else:
            print(">>>>>>>Start finetune Inferencing: {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
            logger.info('>>>>>>>Start finetune Inferencing: {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            mse, mae = exp.Inference()

    elif args.paradigm == 'pretrain_finetune' and args.task_name == "finetune":
        # Start finetuning
        if args.downstream_task == 'forecasting':
            if args.is_training:
                print(">>>>>>>Start finetune training: {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
                logger.info('>>>>>>>Start finetuning training: {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                test_loss_min = exp.finetune()

                print(">>>>>>>Start finetune Inferencing: {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
                logger.info('>>>>>>>Start finetune Inferencing: {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                mse, mae = exp.Inference()

            else:
                print(">>>>>>>Start finetune Inferencing: {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
                logger.info('>>>>>>>Start finetune Inferencing: {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                mse, mae = exp.Inference()

    torch.cuda.empty_cache()

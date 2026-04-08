from torch.utils.data import DataLoader
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pretrain_TimeCAP, Dataset_Solar

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'pretrain_AAAI': Dataset_Pretrain_TimeCAP,
}

def data_provider(args, flag):
    # Ensure consistent configuration for evaluation
    if args.model == 'TimeCAP':
        Data = data_dict[args.pretrain_data if args.task_name == 'pretrain' and flag != 'inference' else args.data]
        size = [args.seq_len, args.label_len, args.pretrain_pred_len] if args.task_name == 'pretrain' and flag != 'inference' else [args.seq_len, args.label_len, args.pred_len]
        shuffle_flag, drop_last = (False, args.drop_last) if flag in ['test', 'inference'] else (True, args.drop_last)
        batch_size = args.pretrain_batch_size if args.task_name == 'pretrain' and flag != 'inference' else args.batch_size

    timeenc = 0 if args.embed != 'timeF' else 1
    data_set = Data(
        args=args,
        flag=flag,
        timeenc=timeenc,
        size=size,
        seasonal_patterns=args.seasonal_patterns  # Monthly
    )

    if args.task_name == 'pretrain' and flag != 'inference':
        data_loader = [
            DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers= args.num_workers,
                drop_last = drop_last,
                pin_memory = True,
            )
            for ds in data_set.datasets
        ]
    else:
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True,
        )

    return data_set, data_loader # 一个Dataset_ETT_hour类型对象，一个DataLoader类型对象
